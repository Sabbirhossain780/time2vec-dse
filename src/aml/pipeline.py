"""Orchestrates the pipeline stages end-to-end, mirroring the original
notebook's block order:

  0. prep-data   : raw DSE workbook -> data/processed/stockprice.csv   (data_prep)
  1. preprocess  : load stockprice.csv -> per-sector sequences/scalers (preprocessing)
  2. train       : per-sector + OVERALL models                        (train)
  3. evaluate    : comparison tables (CSV)                            (evaluate)
  4. visualize   : training curves, prediction plots, result analysis (visualize)
  5. xai         : occlusion sensitivity + integrated gradients       (xai)

Plus an opt-in stage not part of the default `all` run:
  baselines      : naive persistence + ARIMA, for comparison            (baselines)

Each stage is a plain function that can be called independently (e.g. from a
notebook) or chained through `run_all`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import pandas as pd

from . import baselines, data_prep, evaluate, multiseed, preprocessing, train, visualize, xai
from .config import EPOCHS, BATCH_SIZE, MODEL_TYPES, Paths, get_paths, set_seeds

logger = logging.getLogger(__name__)

RUN_STATE_FILENAME = "_run_state.joblib"


class HistoryLike:
    """Stand-in for a Keras History object after round-tripping through storage
    -- only the `.history` dict (used by evaluate/visualize) is needed."""

    def __init__(self, history: dict):
        self.history = history


@dataclass
class PipelineState:
    paths: Paths
    df: Optional[pd.DataFrame] = None
    sectors: List[str] = field(default_factory=list)
    processed_data_dict: Dict = field(default_factory=dict)
    scalers: Dict = field(default_factory=dict)
    all_results: Dict = field(default_factory=dict)
    model_histories: Dict = field(default_factory=dict)
    all_results_overall: Dict = field(default_factory=dict)
    model_histories_overall: Dict = field(default_factory=dict)
    df_per_sector: Optional[pd.DataFrame] = None
    df_overall: Optional[pd.DataFrame] = None


def stage_prep_data(paths: Paths, strict: bool = False) -> pd.DataFrame:
    return data_prep.build_stockprice_csv(paths.raw_xlsx, paths.processed_csv, strict=strict)


def stage_load_and_preprocess(paths: Paths) -> PipelineState:
    state = PipelineState(paths=paths)
    state.df = preprocessing.load_stockprice(paths.processed_csv)
    state.sectors = state.df["sector"].dropna().unique().tolist()
    logger.info("Sectors: %s", state.sectors)
    state.processed_data_dict, state.scalers = preprocessing.process_sectors(
        state.df, state.sectors, paths.scalers_dir)
    return state


def stage_train(state: PipelineState, model_types=MODEL_TYPES, epochs=EPOCHS, batch_size=BATCH_SIZE):
    state.all_results, state.model_histories = train.train_sector_models(
        state.sectors, state.processed_data_dict, state.scalers, state.paths.models_dir,
        model_types=model_types, epochs=epochs, batch_size=batch_size)
    state.all_results_overall, state.model_histories_overall = train.train_overall_models(
        state.sectors, state.processed_data_dict, state.paths.models_dir,
        model_types=model_types, epochs=epochs, batch_size=batch_size)
    save_run_state(state)
    return state


def save_run_state(state: PipelineState) -> None:
    """Persist training histories + evaluation results (predictions, metrics) so that
    evaluate/visualize/xai can run standalone later without retraining."""
    path = state.paths.results_dir / RUN_STATE_FILENAME
    payload = {
        "all_results": state.all_results,
        "model_histories": {m: {s: h.history for s, h in secs.items()}
                             for m, secs in state.model_histories.items()},
        "all_results_overall": state.all_results_overall,
        "model_histories_overall": {m: h.history for m, h in state.model_histories_overall.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    logger.info("Saved run state (histories + results) -> %s", path)


def load_run_state(state: PipelineState) -> bool:
    """Load a previously saved run state into `state`. Returns False if none exists."""
    path = state.paths.results_dir / RUN_STATE_FILENAME
    if not path.exists():
        return False
    payload = joblib.load(path)
    state.all_results = payload["all_results"]
    state.model_histories = {m: {s: HistoryLike(h) for s, h in secs.items()}
                              for m, secs in payload["model_histories"].items()}
    state.all_results_overall = payload["all_results_overall"]
    state.model_histories_overall = {m: HistoryLike(h) for m, h in payload["model_histories_overall"].items()}
    logger.info("Loaded existing run state <- %s", path)
    return True


def stage_evaluate(state: PipelineState, model_types=MODEL_TYPES):
    state.df_per_sector, state.df_overall = evaluate.build_comparison_tables(
        state.sectors, model_types, state.model_histories, state.all_results,
        state.model_histories_overall, state.all_results_overall, state.paths.results_dir)
    return state


def stage_visualize(state: PipelineState, model_types=MODEL_TYPES):
    visualize.generate_training_figures(
        state.sectors, model_types, state.model_histories, state.all_results,
        state.model_histories_overall, state.paths.figures_dir)
    visualize.generate_paper_style_figures(
        state.sectors, model_types, state.processed_data_dict, state.scalers,
        state.paths.models_dir, state.paths.figures_dir)
    if state.df_per_sector is not None:
        visualize.generate_result_analysis_figures(state.df_per_sector, state.paths.figures_dir)
    return state


def stage_xai(state: PipelineState, model_types=MODEL_TYPES):
    return xai.run_xai(state.sectors, model_types, state.processed_data_dict,
                        state.paths.models_dir, state.paths.xai_dir)


def stage_baselines(state: PipelineState):
    """Naive persistence + ARIMA, merged with the trained-model comparison
    table (outputs/results/comparison_per_sector.csv) if it exists."""
    return baselines.run_baselines(state.sectors, state.processed_data_dict,
                                    state.scalers, state.paths.results_dir)


def stage_multiseed_single(state: PipelineState, seed: int, model_types=MODEL_TYPES,
                            epochs=EPOCHS, batch_size=BATCH_SIZE, tag: str = ""):
    """Train one seed's worth of per-sector models and save it durably. Meant
    to be invoked as its own OS process per seed -- see multiseed.py."""
    return multiseed.run_single_seed(
        seed, state.sectors, state.processed_data_dict, state.scalers, state.paths.results_dir,
        model_types=model_types, epochs=epochs, batch_size=batch_size, tag=tag)


def stage_multiseed_merge(state: PipelineState, seeds=None, tag: str = ""):
    """Merge whichever multiseed_seed_<seed><tag>.csv files exist and
    summarize against the (deterministic) naive baseline."""
    df_sweep = multiseed.merge_multiseed_results(state.paths.results_dir, seeds=seeds, tag=tag)
    naive_df = baselines.naive_persistence_baseline(state.sectors, state.processed_data_dict, state.scalers)
    summary = multiseed.summarize_multiseed(df_sweep, naive_df)
    summary_path = state.paths.results_dir / f"multiseed_summary{tag}.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved: %s", summary_path)
    return df_sweep, summary


def run_all(root=None, model_types=MODEL_TYPES, epochs=EPOCHS, batch_size=BATCH_SIZE,
            strict_data_prep: bool = False, skip_xai: bool = False) -> PipelineState:
    paths = get_paths(root)
    set_seeds()

    logger.info("Stage 0/5: prep-data")
    stage_prep_data(paths, strict=strict_data_prep)

    logger.info("Stage 1/5: preprocess")
    state = stage_load_and_preprocess(paths)

    logger.info("Stage 2/5: train")
    stage_train(state, model_types=model_types, epochs=epochs, batch_size=batch_size)

    logger.info("Stage 3/5: evaluate")
    stage_evaluate(state, model_types=model_types)

    logger.info("Stage 4/5: visualize")
    stage_visualize(state, model_types=model_types)

    if not skip_xai:
        logger.info("Stage 5/5: xai")
        stage_xai(state, model_types=model_types)

    logger.info("Pipeline complete. Outputs under %s", paths.root / "outputs")
    return state
