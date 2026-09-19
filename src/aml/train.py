"""Stage 2: train per-sector and pooled OVERALL models, evaluate on the test split.

Mirrors the original notebook's Block 6 (per-sector train/eval/save) and
Block 6b (pooled OVERALL train/eval/save).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import callbacks

from .config import BATCH_SIZE, EARLY_STOP_PATIENCE, EPOCHS, MODEL_TYPES, RET_COLS, TRAIN_RATIO, VAL_RATIO
from .models import build_model_by_name
from .utils import ensure_targets_column, invert_close_only, model_path_for, test_start_index, ts_split

logger = logging.getLogger(__name__)


def _make_early_stop():
    return callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)


def train_sector_models(sectors, processed_data_dict: Dict, scalers: Dict, models_dir: Path,
                         model_types=MODEL_TYPES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         save_models: bool = True):
    """Train every (model_type x sector) combination. Returns (all_results, model_histories).

    save_models=False skips persisting .keras files -- used by the multi-seed
    sweep, which only needs the metrics, not 75 throwaway model files."""
    models_dir = Path(models_dir) if models_dir is not None else None
    all_results = {m: {} for m in model_types}
    model_histories = {m: {} for m in model_types}

    for sector in sectors:
        if sector not in processed_data_dict:
            continue
        logger.info("===== Sector: %s =====", sector)
        pack = processed_data_dict[sector]
        X, y = pack["X"], ensure_targets_column(pack["y"])
        last_closes = pack["last_closes"]
        seq_len, n_feats = X.shape[1], X.shape[2]

        Xtr, ytr, Xv, yv, Xte, yte = ts_split(X, y, TRAIN_RATIO, VAL_RATIO)
        last_closes_test = last_closes[test_start_index(len(y), TRAIN_RATIO, VAL_RATIO):]

        steps_per_epoch = max(1, len(Xtr) // batch_size)

        for model_type in model_types:
            logger.info("--- %s | %s ---", model_type, sector)
            model = build_model_by_name(model_type, seq_len, n_feats,
                                         steps_per_epoch=steps_per_epoch, total_epochs=epochs)
            hist = model.fit(
                Xtr, ytr, validation_data=(Xv, yv),
                epochs=epochs, batch_size=batch_size, verbose=1,
                callbacks=[_make_early_stop()],
            )
            model_histories[model_type][sector] = hist

            test_mse, test_mae = model.evaluate(Xte, yte, verbose=0)

            scaler = scalers[sector]
            y_pred_norm = model.predict(Xte, verbose=0).reshape(-1)
            y_test_norm = yte.reshape(-1)
            y_pred_actual_returns = invert_close_only(scaler, y_pred_norm)
            y_test_actual_returns = invert_close_only(scaler, y_test_norm)

            predicted_prices = last_closes_test + y_pred_actual_returns
            actual_prices = last_closes_test + y_test_actual_returns

            rmse_actual = float(np.sqrt(mean_squared_error(actual_prices, predicted_prices)))
            mae_actual = float(mean_absolute_error(actual_prices, predicted_prices))

            all_results[model_type][sector] = {
                "RMSE_Actual_Return": rmse_actual,
                "MAE_Actual_Return": mae_actual,
                "actual_closing_prices": actual_prices,
                "predicted_closing_prices": predicted_prices,
            }

            if save_models:
                model_path = model_path_for(models_dir, model_type, sector)
                model.save(model_path)
                logger.info("%s saved -> %s | Test MSE=%.6f MAE=%.6f | Price RMSE=%.6f MAE=%.6f",
                            model_type, model_path, test_mse, test_mae, rmse_actual, mae_actual)
            else:
                logger.info("%s | %s | Test MSE=%.6f MAE=%.6f | Price RMSE=%.6f MAE=%.6f",
                            model_type, sector, test_mse, test_mae, rmse_actual, mae_actual)

    return all_results, model_histories


def concat_overall(processed_data_dict: Dict, sectors):
    Xs, ys = [], []
    for s in sectors:
        if s in processed_data_dict:
            Xs.append(processed_data_dict[s]["X"])
            ys.append(ensure_targets_column(processed_data_dict[s]["y"]))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def train_overall_models(sectors, processed_data_dict: Dict, models_dir: Path,
                          model_types=MODEL_TYPES, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train pooled OVERALL models across all sectors. Returns (all_results_overall, model_histories_overall)."""
    models_dir = Path(models_dir)
    X_all, y_all = concat_overall(processed_data_dict, sectors)
    seq_len_all, n_feats_all = X_all.shape[1], X_all.shape[2]
    Xtr_all, ytr_all, Xv_all, yv_all, Xte_all, yte_all = ts_split(X_all, y_all, TRAIN_RATIO, VAL_RATIO)
    logger.info("OVERALL: Train %s, Val %s, Test %s", Xtr_all.shape, Xv_all.shape, Xte_all.shape)

    model_histories_overall = {}
    all_results_overall = {}
    steps_per_epoch_all = max(1, len(Xtr_all) // batch_size)

    for model_type in model_types:
        logger.info("=== OVERALL %s ===", model_type)
        model = build_model_by_name(model_type, seq_len_all, n_feats_all,
                                     steps_per_epoch=steps_per_epoch_all, total_epochs=epochs)
        hist = model.fit(
            Xtr_all, ytr_all, validation_data=(Xv_all, yv_all),
            epochs=epochs, batch_size=batch_size, verbose=1,
            callbacks=[_make_early_stop()],
        )
        model_histories_overall[model_type] = hist

        mse, mae = model.evaluate(Xte_all, yte_all, verbose=0)
        all_results_overall[model_type] = {"Test_MSE": float(mse), "Test_MAE": float(mae)}

        path = model_path_for(models_dir, model_type, "OVERALL")
        model.save(path)
        logger.info("Saved OVERALL -> %s | Test MSE=%.6f MAE=%.6f", path, mse, mae)

    return all_results_overall, model_histories_overall
