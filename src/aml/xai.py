"""Stage 5: explainability -- occlusion sensitivity + integrated gradients.

The original notebooks never contained the code that produced
archive/reference_outputs/results/xai/*.csv (only the output files survived,
plus a summary table with empty IG/SHAP columns and one orphaned partial IG
file). This is a fresh implementation that reproduces the same file layout
and semantics inferred from those outputs:

- Occlusion "heat" is an (seq_len x n_features) matrix where heat[t, f] is the
  mean absolute change in the model's prediction when that single
  (timestep, feature) cell is zeroed out across the test set.
- Occlusion "time"/"feature" are row-sums/column-sums of the heat matrix
  (verified against the archived reference CSVs -- they reproduce exactly).
- Integrated Gradients attributions use a zero baseline, averaged over the
  test set, shape (seq_len x n_features).

SHAP is intentionally not reproduced: no SHAP output ever existed in the
archived results (the summary column was always empty), so there's nothing
to reconstruct against.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import FEATURES, RET_COLS, SEQ_LEN, XAI_IG_STEPS
from .utils import model_path_for, load_model_for_inference, ts_split

logger = logging.getLogger(__name__)

TIME_LABELS = [f"t-{SEQ_LEN - i}" for i in range(SEQ_LEN)]


def _test_split_for_sector(sector, processed_data_dict):
    pack = processed_data_dict[sector]
    X = pack["X"]
    y = pack["y"].reshape(-1, 1) if np.ndim(pack["y"]) == 1 else pack["y"]
    *_, Xte, yte = ts_split(X, y, 0.8, 0.1)
    return Xte


def occlusion_sensitivity(model, X_test: np.ndarray) -> np.ndarray:
    """Return an (seq_len, n_features) heat matrix of mean |prediction change|
    when each single (timestep, feature) cell is zeroed across the test set."""
    seq_len, n_feats = X_test.shape[1], X_test.shape[2]
    baseline_pred = model.predict(X_test, verbose=0).reshape(-1)

    heat = np.zeros((seq_len, n_feats))
    for t in range(seq_len):
        for f in range(n_feats):
            X_occ = X_test.copy()
            X_occ[:, t, f] = 0.0
            occ_pred = model.predict(X_occ, verbose=0).reshape(-1)
            heat[t, f] = np.mean(np.abs(occ_pred - baseline_pred))
    return heat


def integrated_gradients(model, X_test: np.ndarray, steps: int = XAI_IG_STEPS) -> np.ndarray:
    """Mean Integrated Gradients attribution over the test set, zero baseline,
    shape (seq_len, n_features)."""
    X = tf.convert_to_tensor(X_test, dtype=tf.float32)
    baseline = tf.zeros_like(X)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps), (steps, 1, 1, 1))

    # interpolated: (steps, N, seq, feat)
    diff = X - baseline
    interpolated = baseline[None, ...] + alphas * diff[None, ...]
    interpolated = tf.reshape(interpolated, (-1, X.shape[1], X.shape[2]))

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
    grads = tape.gradient(preds, interpolated)
    grads = tf.reshape(grads, (steps, X.shape[0], X.shape[1], X.shape[2]))

    avg_grads = tf.reduce_mean(grads, axis=0)  # (N, seq, feat)
    attributions = diff * avg_grads  # (N, seq, feat)
    return tf.reduce_mean(attributions, axis=0).numpy()  # (seq, feat)


def _save_matrix(matrix: np.ndarray, index, columns, path: Path):
    pd.DataFrame(matrix, index=index, columns=columns).to_csv(path)


def _top_time_feature(heat: np.ndarray):
    time_importance = heat.sum(axis=1)
    feature_importance = heat.sum(axis=0)
    top_time = TIME_LABELS[int(np.argmax(time_importance))]
    top_feat = FEATURES[int(np.argmax(feature_importance))]
    return time_importance, feature_importance, top_time, top_feat


def run_xai(sectors, model_types, processed_data_dict: Dict, models_dir: Path, xai_dir: Path,
            ig_steps: int = XAI_IG_STEPS):
    xai_dir = Path(xai_dir)
    xai_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for model_type in model_types:
        for sector in sectors:
            if sector not in processed_data_dict:
                continue
            path = model_path_for(models_dir, model_type, sector)
            if not path.exists():
                logger.warning("[xai] Missing model: %s", path)
                continue

            model = load_model_for_inference(path)
            Xte = _test_split_for_sector(sector, processed_data_dict)
            base = f"{model_type}_{sector.replace(' ', '_')}"

            heat = occlusion_sensitivity(model, Xte)
            time_imp, feat_imp, top_time_occ, top_feat_occ = _top_time_feature(heat)

            _save_matrix(heat, TIME_LABELS, FEATURES, xai_dir / f"{base}_Occlusion_heat.csv")
            pd.Series(time_imp, index=TIME_LABELS).to_csv(xai_dir / f"{base}_Occlusion_time.csv")
            pd.Series(feat_imp, index=FEATURES).to_csv(xai_dir / f"{base}_Occlusion_feature.csv")

            ig_mean = integrated_gradients(model, Xte, steps=ig_steps)
            _save_matrix(ig_mean, TIME_LABELS, FEATURES, xai_dir / f"{base}_IG_mean.csv")
            ig_time_imp, ig_feat_imp, top_time_ig, top_feat_ig = _top_time_feature(np.abs(ig_mean))

            summary_rows.append({
                "Model": model_type,
                "Sector": sector,
                "TopTime_IG": top_time_ig,
                "TopFeat_IG": top_feat_ig,
                "TopTime_SHAP": np.nan,
                "TopFeat_SHAP": np.nan,
                "TopTime_Occlusion": top_time_occ,
                "TopFeat_Occlusion": top_feat_occ,
            })
            logger.info("[xai] %s | %s done", model_type, sector)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(xai_dir / "xai_summary.csv", index=False)
    logger.info("Saved XAI summary -> %s", xai_dir / "xai_summary.csv")
    return df_summary
