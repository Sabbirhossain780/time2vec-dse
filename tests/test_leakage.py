"""Leakage guards for the preprocessing/split chain.

Targets are first differences of a 10-day moving average, so labels overlap by
construction; these tests pin down that (a) sequences never see the future,
(b) the scaler never sees beyond the training rows, and (c) the split leaves an
embargo gap wide enough to break the label overlap at each boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.config import EMBARGO, RET_COLS, SEQ_LEN, TRAIN_RATIO  # noqa: E402
from aml.preprocessing import create_sequences, make_stationary_and_scale  # noqa: E402
from aml.utils import test_start_index, ts_split  # noqa: E402

FEATURES = ["open", "high", "low", "close", "volume"]


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame({c: walk + rng.normal(0, 0.1, n) for c in FEATURES})


def test_sequences_are_causal():
    """Corrupting held-out rows must leave every training-era sample untouched.

    `cut` sits past the scaler's fit boundary, so the scaler is unaffected too
    -- this checks the end-to-end property (held-out data never reaches a
    training sample), not just the windowing arithmetic."""
    df = _toy()
    cut = 360
    proc_a, _ = make_stationary_and_scale(df)
    Xa, ya, _ = create_sequences(proc_a, SEQ_LEN)

    dirty = df.copy()
    dirty.iloc[cut:] *= 7.0
    proc_b, _ = make_stationary_and_scale(dirty)
    Xb, yb, _ = create_sequences(proc_b, SEQ_LEN)

    # Row index of sample i's target is i + SEQ_LEN; the 10-day MA means a
    # sample is only guaranteed clean if its whole MA window predates `cut`.
    safe = cut - SEQ_LEN - 10
    assert safe > 0
    assert np.allclose(Xa[:safe], Xb[:safe]), "future rows leaked into past sequences"
    assert np.allclose(ya[:safe], yb[:safe]), "future rows leaked into past targets"


def test_scaler_fitted_on_training_rows_only():
    """Changing only the held-out tail must not move the fitted scaler."""
    df = _toy()
    _, scaler_a = make_stationary_and_scale(df)

    n = len(df)
    dirty = df.copy()
    dirty.iloc[int(n * 0.95):] *= 50.0   # deep inside the test block
    _, scaler_b = make_stationary_and_scale(dirty)

    assert np.allclose(scaler_a.data_min_, scaler_b.data_min_), "test rows leaked into scaler"
    assert np.allclose(scaler_a.data_max_, scaler_b.data_max_), "test rows leaked into scaler"


def test_split_embargo_breaks_label_overlap():
    """Gap between blocks must be >= the MA window that makes labels overlap."""
    n = 845
    X = np.arange(n).reshape(n, 1, 1).astype("float32")
    y = np.arange(n).astype("float32")
    Xtr, ytr, Xv, yv, Xte, yte = ts_split(X, y, TRAIN_RATIO, 0.1)

    assert yv[0] - ytr[-1] >= EMBARGO, "train/val boundary has overlapping labels"
    assert yte[0] - yv[-1] >= EMBARGO, "val/test boundary has overlapping labels"
    assert len(yte) == n - int(n * TRAIN_RATIO) - int(n * 0.1), "test window moved"


def test_test_start_index_matches_split():
    """Anything sliced by test_start_index must line up with ts_split's test block."""
    for n in (845, 844, 400, 101):
        X = np.arange(n).reshape(n, 1, 1).astype("float32")
        y = np.arange(n).astype("float32")
        *_, Xte, yte = ts_split(X, y, TRAIN_RATIO, 0.1)
        start = test_start_index(n, TRAIN_RATIO, 0.1)
        assert len(yte) == n - start, f"test block length mismatch at n={n}"
        assert yte[0] == start, f"test block start mismatch at n={n}"


if __name__ == "__main__":
    for fn in (test_sequences_are_causal,
               test_scaler_fitted_on_training_rows_only,
               test_split_embargo_breaks_label_overlap,
               test_test_start_index_matches_split):
        fn()
        print(f"PASS {fn.__name__}")
