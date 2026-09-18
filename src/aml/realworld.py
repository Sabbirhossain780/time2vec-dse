"""Optional stage: evaluate saved models against a fresh, unseen CSV.

Ported from the original notebook's Block 10, which was present but disabled
(wrapped in a triple-quoted string, never run against real data). Exposed
here as an explicit opt-in pipeline stage: `run_pipeline.py realworld --input path.csv`.

Expected input columns: sector, trading date, open, high, low, close, volume.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import FEATURES, RET_COLS, SEQ_LEN
from .utils import invert_close_only, model_path_for, load_model_for_inference

logger = logging.getLogger(__name__)


def _normalize_columns(df_):
    df_ = df_.copy()
    df_.columns = df_.columns.str.strip()
    if "trading date" not in df_.columns:
        for k in ["trading_date", "date", "Trading Date", "Trading date"]:
            if k in df_.columns:
                df_["trading date"] = df_[k]
                break
    df_["trading date"] = pd.to_datetime(df_["trading date"], errors="coerce")
    return df_.dropna(subset=["trading date"]).sort_values(["sector", "trading date"])


def _make_stationary_returns(g):
    g = g.copy()
    smooth = g[FEATURES].rolling(window=10, min_periods=1).mean()
    for c in FEATURES:
        g[f"return_{c}"] = smooth[c].diff()
    return g.dropna().reset_index(drop=True)


def _apply_training_scaler(g, sector_name, scalers_dir: Path):
    pkl = Path(scalers_dir) / f"scaler_{sector_name.replace(' ', '_')}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Saved scaler not found for sector {sector_name}: {pkl}")
    scaler = joblib.load(pkl)
    g[RET_COLS] = scaler.transform(g[RET_COLS])
    return g, scaler


def _create_sequences_for_inference(data_df, seq_len=SEQ_LEN):
    X, last_closes, idx_map = [], [], []
    for i in range(seq_len, len(data_df)):
        X.append(data_df[RET_COLS].iloc[i - seq_len:i].values)
        last_closes.append(data_df["close"].iloc[i - 1])
        idx_map.append(i)
    return np.array(X), np.array(last_closes), np.array(idx_map)


def _plot_realworld(actual_prices, predicted_prices, title, base, fig_dir: Path):
    plt.figure()
    plt.plot(actual_prices, label="Actual"); plt.plot(predicted_prices, label="Predicted")
    plt.title(f"{title} — Real-world")
    plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / f"{base}_realworld.png", dpi=160); plt.close()

    z = min(150, len(actual_prices))
    plt.figure()
    plt.plot(actual_prices[-z:], label="Actual"); plt.plot(predicted_prices[-z:], label="Predicted")
    plt.title(f"{title} — Real-world (zoom last {z})")
    plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / f"{base}_realworld_zoom.png", dpi=160); plt.close()


def run_realworld_eval(real_world_csv: Path, sectors, model_types, models_dir: Path,
                        scalers_dir: Path, results_dir: Path, figures_dir: Path):
    results_dir = Path(results_dir)
    fig_dir = Path(figures_dir) / "realworld"
    fig_dir.mkdir(parents=True, exist_ok=True)

    real_df = _normalize_columns(pd.read_csv(real_world_csv))
    rows = []

    for sector in sectors:
        sec_df = real_df[real_df["sector"] == sector].copy()
        if len(sec_df) < SEQ_LEN + 5:
            logger.warning("[real] Skipping %s: too few rows (%d)", sector, len(sec_df))
            continue

        sec_proc = _make_stationary_returns(sec_df)
        try:
            sec_proc, scaler = _apply_training_scaler(sec_proc, sector, scalers_dir)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        X_inf, last_closes_inf, idx_map = _create_sequences_for_inference(sec_proc)
        y_true_norm = sec_proc["return_close"].iloc[idx_map].values
        y_true_ret = invert_close_only(scaler, y_true_norm)

        for model_type in model_types:
            model_path = model_path_for(models_dir, model_type, sector)
            if not model_path.exists():
                logger.warning("[real] Missing model: %s", model_path)
                continue

            model = load_model_for_inference(model_path)
            y_pred_norm = model.predict(X_inf, verbose=0).reshape(-1)
            y_pred_ret = invert_close_only(scaler, y_pred_norm)

            pred_px = last_closes_inf + y_pred_ret
            true_px = last_closes_inf + y_true_ret
            rmse = float(np.sqrt(mean_squared_error(true_px, pred_px)))
            mae = float(mean_absolute_error(true_px, pred_px))

            rows.append({"Scope": "Per-Sector", "Sector": sector, "Model": model_type,
                         "RMSE_RealPrice": rmse, "MAE_RealPrice": mae, "N": len(true_px)})

            base = f"{model_type}_{sector.replace(' ', '_')}"
            _plot_realworld(true_px, pred_px, f"{model_type} | {sector}", base, fig_dir)

    df_realworld = pd.DataFrame(rows).sort_values(["Scope", "Model", "Sector"])
    out_csv = results_dir / "realworld_comparison.csv"
    df_realworld.to_csv(out_csv, index=False)
    logger.info("Saved real-world comparison: %s", out_csv)
    return df_realworld
