"""Small shared helpers used across training, evaluation, visualization, and XAI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import RET_COLS


def ensure_targets_column(y):
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def ts_split(X, y, train_ratio=0.8, val_ratio=0.1):
    n = X.shape[0]
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = max(1, n - n_train - n_val)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xv, yv = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    Xte, yte = X[n_train + n_val:], y[n_train + n_val:]
    return Xtr, ytr, Xv, yv, Xte, yte


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
