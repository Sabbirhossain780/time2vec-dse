"""Stage 1: load stockprice.csv and turn it into per-sector model-ready sequences.

Mirrors the original notebook's Block 2 (load/inspect) and Block 4
(stationarity + scaling + sequence windows).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import FEATURES, RET_COLS, SEQ_LEN, TRAIN_RATIO

logger = logging.getLogger(__name__)

DATE_COLUMN_ALIASES = ["trading_date", "trading date", "date", "Trading Date", "Trading date"]


def load_stockprice(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "trading date" not in df.columns:
        for alt in DATE_COLUMN_ALIASES:
            if alt in df.columns:
                df["trading date"] = df[alt]
                break

    df["trading date"] = pd.to_datetime(df["trading date"], errors="coerce")
    df = df.dropna(subset=["trading date"]).sort_values(["sector", "trading date"]).reset_index(drop=True)
    return df


def make_stationary_and_scale(group: pd.DataFrame, features=FEATURES, ret_cols=RET_COLS,
                               seq_len: int = SEQ_LEN, train_ratio: float = TRAIN_RATIO):
    """10-day MA smoothing -> differencing (returns) -> dropna -> MinMax scale on returns.

    The scaler is fitted on the training rows only. Fitting it on the whole
    series (as the original notebook did) lets the validation and test minima
    and maxima set the scale that the training data is normalized by, which is
    a mild but real form of look-ahead."""
    g = group.copy()
    smooth = g[features].rolling(window=10, min_periods=1).mean()
    for c in features:
        g[f"return_{c}"] = smooth[c].diff()
    g = g.dropna().reset_index(drop=True)

    # Sequence i predicts row i + seq_len, so the last training row is the
    # target of the last training sequence.
    n_seq = max(1, len(g) - seq_len)
    n_fit = min(len(g), seq_len + max(1, int(n_seq * train_ratio)))

    scaler = MinMaxScaler()
    scaler.fit(g[ret_cols].iloc[:n_fit])
    g[ret_cols] = scaler.transform(g[ret_cols])
    return g, scaler


def create_sequences(data: pd.DataFrame, seq_len: int = SEQ_LEN, ret_cols=RET_COLS):
    """Build (X, y, last_close): X uses previous seq_len returns; y is next-step return_close."""
    X, y, last_closes = [], [], []
    for i in range(seq_len, len(data)):
        X.append(data[ret_cols].iloc[i - seq_len:i].values)
        y.append(data["return_close"].iloc[i])
        last_closes.append(data["close"].iloc[i - 1])
    return np.array(X), np.array(y), np.array(last_closes)


def process_sectors(df: pd.DataFrame, sectors, scalers_dir: Path,
                     seq_len: int = SEQ_LEN, features=FEATURES, ret_cols=RET_COLS) -> tuple[Dict, Dict]:
    """Process every sector -> {sector: {X, y, last_closes, raw_data}}, save fitted scalers."""
    scalers_dir = Path(scalers_dir)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    scalers = {}
    processed_data_dict = {}

    for s in sectors:
        sub = df[df["sector"] == s].copy()
        if len(sub) < seq_len + 5:
            logger.warning("Skipping sector %s: too few rows.", s)
            continue
        processed, scaler = make_stationary_and_scale(sub, features, ret_cols, seq_len=seq_len)
        X, y, last_closes = create_sequences(processed, seq_len, ret_cols)

        processed_data_dict[s] = {"X": X, "y": y, "last_closes": last_closes, "raw_data": processed}
        scalers[s] = scaler
        joblib.dump(scaler, scalers_dir / f"scaler_{s.replace(' ', '_')}.pkl")
        logger.info("%s: X%s, y%s, last_closes%s", s, X.shape, y.shape, last_closes.shape)

    return processed_data_dict, scalers
