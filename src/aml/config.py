"""Central configuration: paths, hyperparameters, and constants for the pipeline.

Mirrors the original notebook's Block 1 configuration, but rooted at the
project directory instead of a hardcoded Google Drive path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = two levels up from this file (src/aml/config.py -> AML/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Modeling hyperparameters (from the original notebook, Blocks 1/4/6) ---
SEQ_LEN = 8
FEATURES = ["open", "high", "low", "close", "volume"]
RET_COLS = [f"return_{c}" for c in FEATURES]
MODEL_TYPES = ["Transformer", "LSTM", "RNN"]

EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 5
RANDOM_SEED = 42

# The 5 sector codes kept in the raw DSE workbook, and how they map to the
# clean sector names used everywhere downstream (matches the provenance note
# in data/raw/DSE_master_filtered.xlsx).
RAW_SECTOR_CODE_MAP = {
    "04.Engineering": "Engineering",
    "07.Fuel_&_Power": "Fuel & Power",
    "09.IT_Sector": "IT Sector",
    "15.Services_&_Real_Estate": "Services & Real Estate",
    "17.Telecommunication": "Telecommunication",
}
EXPECTED_ROW_COUNT = 4270  # from data/raw/DSE_master_filtered.xlsx provenance note

# XAI hyperparameters
XAI_IG_STEPS = 50


@dataclass
class Paths:
    root: Path

    raw_xlsx: Path = field(init=False)
    processed_csv: Path = field(init=False)

    figures_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    scalers_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    xai_dir: Path = field(init=False)

    def __post_init__(self):
        self.raw_xlsx = self.root / "data" / "raw" / "DSE-2017 to 2021.csv.xlsx"
        self.processed_csv = self.root / "data" / "processed" / "stockprice.csv"

        self.figures_dir = self.root / "outputs" / "figures"
        self.models_dir = self.root / "outputs" / "models"
        self.scalers_dir = self.models_dir / "scalers"
        self.results_dir = self.root / "outputs" / "results"
        self.xai_dir = self.results_dir / "xai"

    def ensure_dirs(self):
        for d in (self.figures_dir, self.models_dir, self.scalers_dir,
                  self.results_dir, self.xai_dir, self.processed_csv.parent):
            d.mkdir(parents=True, exist_ok=True)


def get_paths(root: str | Path | None = None) -> Paths:
    p = Paths(root=Path(root) if root is not None else PROJECT_ROOT)
    p.ensure_dirs()
    return p


def set_seeds(seed: int = RANDOM_SEED):
    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)
