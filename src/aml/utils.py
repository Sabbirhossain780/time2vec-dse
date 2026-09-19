"""Small shared helpers used across training, evaluation, visualization, and XAI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import EMBARGO, RET_COLS


def ensure_targets_column(y):
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def ts_split(X, y, train_ratio=0.8, val_ratio=0.1, embargo=EMBARGO):
    """Chronological 3-way split with an embargo gap at each boundary.

    Split points are computed on the full length so the test window matches
    the un-embargoed version; the embargo is taken out of the *end* of the
    train and validation blocks, discarding samples whose labels overlap the
    next block's."""
    n = X.shape[0]
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    tr_end = max(1, n_train - embargo)
    v_start, v_end = n_train, max(n_train + 1, n_train + n_val - embargo)
    te_start = n_train + n_val

    Xtr, ytr = X[:tr_end], y[:tr_end]
    Xv, yv = X[v_start:v_end], y[v_start:v_end]
    Xte, yte = X[te_start:], y[te_start:]
    return Xtr, ytr, Xv, yv, Xte, yte


def test_start_index(n: int, train_ratio=0.8, val_ratio=0.1) -> int:
    """Index where the test block begins, matching ts_split.

    Deliberately independent of the embargo: the embargo shortens the train
    and validation blocks, it never moves the test window. Anything aligned
    to the sample axis (e.g. `last_closes`) must be sliced with this rather
    than inferred from len(train) + len(val)."""
    return max(1, int(n * train_ratio)) + max(1, int(n * val_ratio))


def invert_close_only(scaler, arr_norm_1d, ret_cols=RET_COLS):
    """Invert only the close-return dimension of a fitted MinMaxScaler back to real units."""
    close_idx = ret_cols.index("return_close")
    dummy = np.zeros((len(arr_norm_1d), len(ret_cols)))
    dummy[:, close_idx] = arr_norm_1d
    inv = scaler.inverse_transform(dummy)
    return inv[:, close_idx]


def model_path_for(models_dir: Path, model_type: str, sector: str) -> Path:
    return Path(models_dir) / f"{model_type}_model_{sector.replace(' ', '_')}.keras"


def load_model_for_inference(path: Path):
    """Load a saved .keras model for prediction. safe_mode=False is required because
    the Transformer head uses a Lambda layer (take_last) with a Python lambda."""
    m = tf.keras.models.load_model(path, compile=False, safe_mode=False)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m
