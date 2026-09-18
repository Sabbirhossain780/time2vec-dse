#!/usr/bin/env python
"""CLI entrypoint for the DSE RNN/LSTM/Transformer pipeline.

Examples
--------
Run everything, from raw data to XAI:
    python run_pipeline.py all

Run a fast smoke test (1 epoch, skips XAI) to check the wiring works:
    python run_pipeline.py all --smoke-test

Run a single stage (each depends on the previous stage's outputs already existing):
    python run_pipeline.py prep-data
    python run_pipeline.py train --epochs 20
    python run_pipeline.py evaluate
    python run_pipeline.py visualize
    python run_pipeline.py xai

Evaluate saved models against a fresh, unseen CSV:
    python run_pipeline.py realworld --input path/to/new_data.csv

Classical baselines (naive persistence + ARIMA), merged with the trained-model
comparison table if it already exists:
    python run_pipeline.py baselines

Multi-seed robustness sweep -- run ONE seed per process invocation (each
seed saves durably to outputs/results/multiseed_seed_<seed>.csv immediately,
so a crash on one seed doesn't lose the others), then merge:
    python run_pipeline.py multiseed --seed 42
    python run_pipeline.py multiseed --seed 7
    python run_pipeline.py multiseed --seed 123
    python run_pipeline.py multiseed --seed 2024
    python run_pipeline.py multiseed --seed 8675309
    python run_pipeline.py multiseed-merge
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aml import pipeline, realworld  # noqa: E402
from aml.config import BATCH_SIZE, EPOCHS, MODEL_TYPES, get_paths, set_seeds  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[
        "all", "prep-data", "preprocess", "train", "evaluate", "visualize", "xai", "realworld",
        "baselines", "multiseed", "multiseed-merge",
    ])
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for the 'multiseed' stage (one seed per invocation).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Seeds for 'multiseed-merge' (default: 42 7 123 2024 8675309).")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--models", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES,
                         help="Subset of model types to run (default: all three).")
    parser.add_argument("--strict-data-prep", action="store_true",
                         help="Fail if the reconstructed stockprice.csv row count doesn't match the documented provenance (4270).")
    parser.add_argument("--skip-xai", action="store_true", help="Skip the XAI stage when running 'all'.")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Fast end-to-end check: 1 epoch, skips XAI. Use to verify the pipeline wiring, not for real results.")
    parser.add_argument("--input", type=str, help="Input CSV for the 'realworld' stage.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    epochs = 1 if args.smoke_test else args.epochs
    skip_xai = args.skip_xai or args.smoke_test

    if args.stage == "all":
        pipeline.run_all(model_types=args.models, epochs=epochs, batch_size=args.batch_size,
                          strict_data_prep=args.strict_data_prep, skip_xai=skip_xai)
        return

    paths = get_paths()
    set_seeds()

    if args.stage == "prep-data":
        pipeline.stage_prep_data(paths, strict=args.strict_data_prep)
        return

    if args.stage == "preprocess":
        pipeline.stage_load_and_preprocess(paths)
        return

    # Stages below need the full state (df, processed sequences, scalers).
    state = pipeline.stage_load_and_preprocess(paths)

    if args.stage == "train":
        pipeline.stage_train(state, model_types=args.models, epochs=epochs, batch_size=args.batch_size)
    elif args.stage in ("evaluate", "visualize"):
        if not pipeline.load_run_state(state):
            logging.getLogger(__name__).info("No saved run state found; training first.")
            pipeline.stage_train(state, model_types=args.models, epochs=epochs, batch_size=args.batch_size)
        pipeline.stage_evaluate(state, model_types=args.models)
        if args.stage == "visualize":
            pipeline.stage_visualize(state, model_types=args.models)
    elif args.stage == "xai":
        pipeline.stage_xai(state, model_types=args.models)
    elif args.stage == "realworld":
        if not args.input:
            parser.error("--input is required for the 'realworld' stage")
        realworld.run_realworld_eval(
            args.input, state.sectors, args.models, paths.models_dir,
            paths.scalers_dir, paths.results_dir, paths.figures_dir)
    elif args.stage == "baselines":
        pipeline.stage_baselines(state)
    elif args.stage == "multiseed":
        if args.seed is None:
            parser.error("--seed is required for the 'multiseed' stage (one seed per invocation)")
        pipeline.stage_multiseed_single(state, seed=args.seed, model_types=args.models,
                                         epochs=epochs, batch_size=args.batch_size)
    elif args.stage == "multiseed-merge":
        pipeline.stage_multiseed_merge(state, seeds=args.seeds)


if __name__ == "__main__":
    main()
