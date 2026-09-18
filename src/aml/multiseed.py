"""Multi-seed robustness sweep: retrain RNN/LSTM/Transformer per sector under
several random weight initializations, to check whether the single-seed
"beats naive" verdicts and model ranking reported in the README actually
hold, or were an artifact of one lucky/unlucky initialization.

Per-sector models only (not OVERALL) -- the question this answers is
specifically about the per-sector comparison table, and skipping OVERALL
roughly halves the sweep's runtime. No models are persisted to disk; only
metrics are collected (see train.train_sector_models(save_models=False)).

ARIMA and the naive baseline are deterministic given the data -- they don't
need to be swept.

Each seed is meant to be run as its own OS process (see run_pipeline.py's
`multiseed` stage: `--seed <one int>`), not looped over inside one long-lived
Python process. Two reasons: (1) each seed's result is written to its own
outputs/results/multiseed_seed_<seed>.csv immediately, so a crash on seed N
doesn't lose seeds 1..N-1; (2) a long-lived process training dozens of fresh
Keras models back-to-back has been observed here to hit a pathological XLA
JIT recompilation stall (9+ minutes on a single compile) partway through --
a fresh process per seed avoids whatever compile-cache state degrades across
many model.fit() calls in one process.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import BATCH_SIZE, EPOCHS, MODEL_TYPES, set_seeds
from .train import train_sector_models

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 7, 123, 2024, 8675309]


def run_single_seed(seed: int, sectors, processed_data_dict: Dict, scalers: Dict, results_dir: Path,
                     model_types=MODEL_TYPES, epochs=EPOCHS, batch_size=BATCH_SIZE) -> pd.DataFrame:
    """Train every (model_type x sector) combination for one seed and save
    immediately to outputs/results/multiseed_seed_<seed>.csv."""
    results_dir = Path(results_dir)
    logger.info("===== Seed %s =====", seed)
    set_seeds(seed)
    all_results, _ = train_sector_models(
        sectors, processed_data_dict, scalers, models_dir=None,
        model_types=model_types, epochs=epochs, batch_size=batch_size,
        save_models=False)

    rows = []
    for model_type in model_types:
        for sector in sectors:
            res = all_results.get(model_type, {}).get(sector)
            if res is None:
                continue
            rows.append({
                "Seed": seed, "Model": model_type, "Sector": sector,
                "Test_RMSE_Actual_Return": res["RMSE_Actual_Return"],
                "Test_MAE_Actual_Return": res["MAE_Actual_Return"],
            })

    df = pd.DataFrame(rows)
    out_path = results_dir / f"multiseed_seed_{seed}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s (%d rows)", out_path, len(df))
    return df


def merge_multiseed_results(results_dir: Path, seeds: List[int] = None) -> pd.DataFrame:
    """Concatenate whichever outputs/results/multiseed_seed_*.csv files exist
    into the combined outputs/results/multiseed_comparison.csv."""
    results_dir = Path(results_dir)
    seeds = seeds or DEFAULT_SEEDS

    frames = []
    missing = []
    for seed in seeds:
        p = results_dir / f"multiseed_seed_{seed}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            missing.append(seed)
    if missing:
        logger.warning("Missing per-seed files for seeds %s -- merging what's available.", missing)
    if not frames:
        raise FileNotFoundError(f"No multiseed_seed_*.csv files found under {results_dir}")

    df = pd.concat(frames, ignore_index=True)
    out_path = results_dir / "multiseed_comparison.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s (%d seeds merged, %d rows)", out_path, len(frames), len(df))
    return df


def summarize_multiseed(df: pd.DataFrame, naive_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std RMSE per (model, sector) across seeds, plus how many of the
    seeds beat the (deterministic) naive baseline for that sector."""
    naive_lookup = naive_df.set_index("Sector")["Test_RMSE_Actual_Return"].to_dict()

    summary = (df.groupby(["Model", "Sector"])["Test_RMSE_Actual_Return"]
               .agg(["mean", "std", "min", "max", "count"])
               .rename(columns={"count": "n_seeds"})
               .reset_index())
    summary["Naive_RMSE"] = summary["Sector"].map(naive_lookup)

    def _beats_fraction(row):
        sub = df[(df["Model"] == row["Model"]) & (df["Sector"] == row["Sector"])]
        naive_rmse = naive_lookup.get(row["Sector"])
        if naive_rmse is None:
            return None
        return float((sub["Test_RMSE_Actual_Return"] < naive_rmse).mean())

    summary["Frac_seeds_beat_naive"] = summary.apply(_beats_fraction, axis=1)
    return summary.sort_values(["Sector", "mean"])
