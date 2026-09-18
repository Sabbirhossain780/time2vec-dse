"""Classical baselines -- naive persistence and ARIMA -- for comparison
against the trained RNN/LSTM/Transformer models.

Both baselines are evaluated on exactly the same chronological test window,
scaler, and price-reconstruction path as the trained models (see
src/aml/train.py), so RMSE/MAE here are directly comparable to the
Test_RMSE_Actual_Return / Test_MAE_Actual_Return columns in
outputs/results/comparison_per_sector.csv.

ARIMA methodology: order (p, d, q) is selected per sector by AIC grid search
on the train+val portion (mirroring how the neural nets also use val data,
via early stopping, before ever touching test), then evaluated with
walk-forward one-step-ahead forecasts through the test window -- the
standard way to evaluate a classical time-series model, and a fairer test of
ARIMA than a single static multi-step forecast would be.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from .config import SEQ_LEN, TRAIN_RATIO, VAL_RATIO
from .utils import ensure_targets_column, invert_close_only

logger = logging.getLogger(__name__)

ARIMA_P_RANGE = range(0, 4)
ARIMA_D_RANGE = (0, 1)
ARIMA_Q_RANGE = range(0, 4)


def _sector_split_indices(n: int, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = max(1, n - n_train - n_val)
    return n_train, n_val


def naive_persistence_baseline(sectors, processed_data_dict: Dict, scalers: Dict) -> pd.DataFrame:
    """'Tomorrow's close == today's close' -- i.e. predicted return = 0."""
    rows = []
    for s in sectors:
        if s not in processed_data_dict:
            continue
        pack = processed_data_dict[s]
        y = ensure_targets_column(pack["y"]).reshape(-1)
        n_train, n_val = _sector_split_indices(len(y))
        last_closes_test = pack["last_closes"][n_train + n_val:]

        y_test_norm = y[n_train + n_val:]
        scaler = scalers[s]
        y_test_ret = invert_close_only(scaler, y_test_norm)
        actual_prices = last_closes_test + y_test_ret
        pred_prices = last_closes_test  # predicted return = 0

        rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
        mae = float(mean_absolute_error(actual_prices, pred_prices))
        rows.append({"Model": "Naive", "Sector": s, "Test_RMSE_Actual_Return": rmse,
                     "Test_MAE_Actual_Return": mae, "N_test": len(actual_prices)})
    return pd.DataFrame(rows)


def _select_arima_order(train_series: pd.Series):
    best_aic = np.inf
    best_order = None
    for d in ARIMA_D_RANGE:
        for p in ARIMA_P_RANGE:
            for q in ARIMA_Q_RANGE:
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = ARIMA(train_series, order=(p, d, q)).fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    if best_order is None:
        best_order = (1, 0, 0)
    return best_order, best_aic


def _arima_walk_forward(fit_series: pd.Series, test_series: pd.Series, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(fit_series, order=order).fit()

    preds = []
    current = fitted
    for true_val in test_series.values:
        fc = current.get_forecast(steps=1).predicted_mean.iloc[0]
        preds.append(fc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current = current.append([true_val], refit=False)
    return np.array(preds)


def arima_baseline(sectors, processed_data_dict: Dict, scalers: Dict, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    rows = []
    for s in sectors:
        if s not in processed_data_dict:
            continue
        pack = processed_data_dict[s]
        y = ensure_targets_column(pack["y"]).reshape(-1)
        n = len(y)
        n_train, n_val = _sector_split_indices(n)

        # The return_close series aligned 1:1 with y (see create_sequences in
        # preprocessing.py: y[k] == raw_data['return_close'].iloc[seq_len+k]).
        full_series = pack["raw_data"]["return_close"].iloc[seq_len:seq_len + n].reset_index(drop=True)

        fit_series = full_series.iloc[:n_train + n_val]
        test_series = full_series.iloc[n_train + n_val:]

        order, aic = _select_arima_order(fit_series)
        logger.info("[ARIMA] %s: order=%s AIC=%.2f", s, order, aic)

        preds_norm = _arima_walk_forward(fit_series, test_series, order)
        true_norm = test_series.values

        scaler = scalers[s]
        pred_ret = invert_close_only(scaler, preds_norm)
        true_ret = invert_close_only(scaler, true_norm)

        last_closes_test = pack["last_closes"][n_train + n_val:]
        pred_prices = last_closes_test + pred_ret
        actual_prices = last_closes_test + true_ret

        rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
        mae = float(mean_absolute_error(actual_prices, pred_prices))

        rows.append({"Model": "ARIMA", "Sector": s, "Order": str(order),
                     "Test_RMSE_Actual_Return": rmse, "Test_MAE_Actual_Return": mae,
                     "N_test": len(actual_prices)})
    return pd.DataFrame(rows)


def run_baselines(sectors, processed_data_dict: Dict, scalers: Dict, results_dir: Path) -> pd.DataFrame:
    """Compute Naive + ARIMA, merge with the trained-model comparison table if
    it already exists, and save one combined CSV."""
    results_dir = Path(results_dir)

    df_naive = naive_persistence_baseline(sectors, processed_data_dict, scalers)
    df_arima = arima_baseline(sectors, processed_data_dict, scalers)
    df_baselines = pd.concat([df_naive, df_arima], ignore_index=True)

    per_sector_path = results_dir / "comparison_per_sector.csv"
    if per_sector_path.exists():
        df_models = pd.read_csv(per_sector_path)[
            ["Model", "Sector", "Test_RMSE_Actual_Return", "Test_MAE_Actual_Return"]]
        combined = pd.concat([df_baselines, df_models], ignore_index=True)
    else:
        logger.warning("%s not found -- saving baselines only, without the trained models.", per_sector_path)
        combined = df_baselines

    out_path = results_dir / "baselines_comparison.csv"
    combined.sort_values(["Sector", "Model"]).to_csv(out_path, index=False)
    logger.info("Saved: %s", out_path)
    return combined
