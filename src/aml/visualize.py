"""Stage 4: figures -- training curves, prediction plots, sector bar charts,
paper-style processed-vs-raw plots, and cross-sector result-analysis charts.

Mirrors the original notebook's Block 8, Block 11, and the "Result Analysis"
cells added in the Copy of RNN_LSTM_TR.ipynb variant.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RET_COLS
from .utils import test_start_index, invert_close_only, load_model_for_inference, model_path_for, ts_split

logger = logging.getLogger(__name__)


# --- Block 8: training curves, prediction plots, sector bars ---------------

def plot_loss_mae(history, title_prefix, base_name, figures_dir: Path):
    plt.figure()
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title(f"{title_prefix} — Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(Path(figures_dir) / f"{base_name}_loss.png", dpi=160); plt.close()

    plt.figure()
    plt.plot(history.history.get("mae", []), label="Train MAE")
    plt.plot(history.history.get("val_mae", []), label="Val MAE")
    plt.title(f"{title_prefix} — MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend(); plt.tight_layout()
    plt.savefig(Path(figures_dir) / f"{base_name}_mae.png", dpi=160); plt.close()


def plot_pred_vs_actual(actual, pred, title_prefix, base_name, figures_dir: Path, num_points=None):
    a = actual if num_points is None else actual[-num_points:]
    p = pred if num_points is None else pred[-num_points:]
    plt.figure()
    plt.plot(a, label="Actual"); plt.plot(p, label="Predicted")
    plt.title(f"{title_prefix} — Prediction vs Actual")
    plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(Path(figures_dir) / f"{base_name}_pred_vs_actual.png", dpi=160); plt.close()


def plot_zoom(actual, pred, title_prefix, base_name, figures_dir: Path, start=-150, end=None):
    a = actual[start:end]; p = pred[start:end]
    plt.figure()
    plt.plot(a, label="Actual"); plt.plot(p, label="Predicted")
    plt.title(f"{title_prefix} — Zoom")
    plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(Path(figures_dir) / f"{base_name}_zoom.png", dpi=160); plt.close()


def bar_by_sector(metric_name, model_type, results_dict, suffix, sectors, figures_dir: Path):
    sectors_list, vals = [], []
    for s in sectors:
        if s in results_dict.get(model_type, {}):
            sectors_list.append(s)
            vals.append(results_dict[model_type][s][metric_name])
    if not sectors_list:
        return
    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), sectors_list, rotation=45, ha="right")
    plt.ylabel(metric_name.replace("_", " ")); plt.title(f"{model_type} — {metric_name} across sectors")
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / f"{model_type}_{suffix}_{metric_name}.png", dpi=160); plt.close()


def generate_training_figures(sectors, model_types, model_histories, all_results,
                               model_histories_overall, figures_dir: Path):
    figures_dir = Path(figures_dir)
    for model_type in model_types:
        for s in sectors:
            hist = model_histories.get(model_type, {}).get(s)
            if hist is None:
                continue
            base = f"{model_type}_{s.replace(' ', '_')}"
            plot_loss_mae(hist, f"{model_type} | {s}", base, figures_dir)

            res = all_results.get(model_type, {}).get(s, {})
            predicted_prices = res.get("predicted_closing_prices")
            actual_prices = res.get("actual_closing_prices")
            if predicted_prices is not None and actual_prices is not None and len(predicted_prices) > 0:
                plot_pred_vs_actual(actual_prices, predicted_prices, f"{model_type} | {s}", base, figures_dir)
                plot_zoom(actual_prices, predicted_prices, f"{model_type} | {s}", base, figures_dir)

    for model_type, hist in model_histories_overall.items():
        base = f"{model_type}_OVERALL"
        plot_loss_mae(hist, f"{model_type} | OVERALL", base, figures_dir)

    for model_type in model_types:
        bar_by_sector("RMSE_Actual_Return", model_type, all_results, "Test", sectors, figures_dir)
        bar_by_sector("MAE_Actual_Return", model_type, all_results, "Test", sectors, figures_dir)

    logger.info("Saved training/prediction figures under %s", figures_dir)


# --- Block 11: paper-style processed(returns) vs raw(price) plots ----------

def _sector_split_again(sector, processed_data_dict, scalers):
    pack = processed_data_dict[sector]
    X = pack["X"]
    y = pack["y"].reshape(-1, 1) if np.ndim(pack["y"]) == 1 else pack["y"]
    Xtr, ytr, Xv, yv, Xte, yte = ts_split(X, y, 0.8, 0.1)
    last_closes_test = pack["last_closes"][test_start_index(len(y)):]
    return Xtr, ytr, Xv, yv, Xte, yte, last_closes_test, scalers[sector]


def generate_paper_style_figures(sectors, model_types, processed_data_dict, scalers,
                                  models_dir: Path, figures_dir: Path):
    paper_dir = Path(figures_dir) / "paper_style"
    paper_dir.mkdir(parents=True, exist_ok=True)

    for model_type in model_types:
        for sector in sectors:
            if sector not in processed_data_dict:
                continue
            path = model_path_for(models_dir, model_type, sector)
            if not path.exists():
                logger.warning("Missing model file: %s", path)
                continue
            model = load_model_for_inference(path)

            Xtr, ytr, Xv, yv, Xte, yte, last_closes_test, scaler = _sector_split_again(
                sector, processed_data_dict, scalers)

            y_pred_norm = model.predict(Xte, verbose=0).reshape(-1)
            y_true_norm = yte.reshape(-1)
            base = f"{model_type}_{sector.replace(' ', '_')}"

            _plot_processed_returns(y_true_norm, y_pred_norm, f"{model_type} | {sector}", base, paper_dir)

            y_pred_ret = invert_close_only(scaler, y_pred_norm)
            y_true_ret = invert_close_only(scaler, y_true_norm)
            pred_px = last_closes_test + y_pred_ret
            true_px = last_closes_test + y_true_ret
            _plot_raw_prices(true_px, pred_px, f"{model_type} | {sector}", base, paper_dir)

    logger.info("Saved paper-style figures under %s", paper_dir)


def _plot_processed_returns(y_true_norm, y_pred_norm, title_prefix, base, paper_dir: Path):
    plt.figure()
    plt.plot(y_true_norm, label="True (processed)"); plt.plot(y_pred_norm, label="Pred (processed)")
    plt.title(f"{title_prefix} — Processed (returns)")
    plt.xlabel("Test index"); plt.ylabel("Normalized return"); plt.legend(); plt.tight_layout()
    plt.savefig(paper_dir / f"{base}_processed_returns.png", dpi=170); plt.close()

    z = min(150, len(y_true_norm))
    plt.figure()
    plt.plot(y_true_norm[-z:], label="True (processed)"); plt.plot(y_pred_norm[-z:], label="Pred (processed)")
    plt.title(f"{title_prefix} — Processed (returns) zoom last {z}")
    plt.xlabel("Test index (tail)"); plt.ylabel("Normalized return"); plt.legend(); plt.tight_layout()
    plt.savefig(paper_dir / f"{base}_processed_returns_zoom.png", dpi=170); plt.close()


def _plot_raw_prices(actual_px, pred_px, title_prefix, base, paper_dir: Path):
    plt.figure()
    plt.plot(actual_px, label="Actual price"); plt.plot(pred_px, label="Predicted price")
    plt.title(f"{title_prefix} — Raw closing price")
    plt.xlabel("Test index"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(paper_dir / f"{base}_raw_prices.png", dpi=170); plt.close()

    z = min(150, len(actual_px))
    plt.figure()
    plt.plot(actual_px[-z:], label="Actual price"); plt.plot(pred_px[-z:], label="Predicted price")
    plt.title(f"{title_prefix} — Raw price (zoom last {z})")
    plt.xlabel("Test index (tail)"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(paper_dir / f"{base}_raw_prices_zoom.png", dpi=170); plt.close()


# --- Result Analysis (from the "Copy of" notebook variant) -----------------

def generate_result_analysis_figures(df_per_sector: pd.DataFrame, figures_dir: Path):
    """Cross-sector, cross-model comparison bar charts (seaborn), ported from the
    Result Analysis cells added in All/Copy of RNN_LSTM_TR.ipynb."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed; skipping result-analysis figures.")
        return

    figures_dir = Path(figures_dir)
    sns.set_style("whitegrid")

    df_melted = df_per_sector.melt(
        id_vars=["Model", "Sector"],
        value_vars=["Train_MSE_last", "Val_MSE_last", "Test_RMSE_Actual_Return"],
        var_name="Metric", value_name="RMSE",
    )
    df_melted["Metric"] = (df_melted["Metric"]
                            .str.replace("_MSE_last", " MSE (last)")
                            .str.replace("_RMSE_Actual_Return", " RMSE (Actual Return)")
                            .str.replace("_", " "))
    plt.figure(figsize=(16, 8))
    palette = sns.color_palette("viridis", n_colors=df_melted["Metric"].nunique())
    sns.barplot(data=df_melted, x="Sector", y="RMSE", hue="Metric", palette=palette)
    plt.title("Model Performance (RMSE) Across Sectors", fontsize=16)
    plt.xlabel("Sector"); plt.ylabel("RMSE"); plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric", loc="upper right"); plt.tight_layout()
    plt.savefig(figures_dir / "combined_rmse_per_sector.png", dpi=300); plt.close()

    df_melted_tv = df_per_sector.melt(
        id_vars=["Model", "Sector"], value_vars=["Train_MSE_last", "Val_MSE_last"],
        var_name="Metric", value_name="MSE",
    )
    df_melted_tv["Metric"] = df_melted_tv["Metric"].str.replace("_MSE_last", " MSE (last)").str.replace("_", " ")
    plt.figure(figsize=(16, 8))
    palette = sns.color_palette("viridis", n_colors=df_melted_tv["Metric"].nunique())
    sns.barplot(data=df_melted_tv, x="Sector", y="MSE", hue="Metric", palette=palette)
    plt.title("Model Performance (Train and Validation MSE) Across Sectors", fontsize=16)
    plt.xlabel("Sector"); plt.ylabel("MSE"); plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric", loc="upper right"); plt.tight_layout()
    plt.savefig(figures_dir / "combined_train_val_mse_per_sector.png", dpi=300); plt.close()

    df_melted_all = df_per_sector.melt(
        id_vars=["Model", "Sector"],
        value_vars=["Train_MSE_last", "Val_MSE_last", "Test_RMSE_Actual_Return",
                    "Train_MAE_last", "Val_MAE_last", "Test_MAE_Actual_Return"],
        var_name="Metric", value_name="Value",
    )
    df_melted_all["Metric_Type"] = df_melted_all["Metric"].apply(
        lambda x: "RMSE" if "MSE" in x or "RMSE" in x else "MAE")
    df_melted_all["Data_Split"] = df_melted_all["Metric"].apply(
        lambda x: "Train" if "Train" in x else ("Val" if "Val" in x else "Test"))
    g = sns.catplot(
        data=df_melted_all, x="Sector", y="Value", hue="Model",
        col="Data_Split", row="Metric_Type", kind="bar",
        height=4, aspect=.7, palette="viridis", sharey="row",
    )
    g.fig.suptitle("Model Performance Across Sectors (RMSE and MAE)", y=1.02, fontsize=16)
    g.set_axis_labels("Sector", "Value")
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{row_name} {col_name}")
    g.tight_layout()
    g.savefig(figures_dir / "combined_all_metrics_per_sector_faceted.png", dpi=300)
    plt.close("all")

    logger.info("Saved result-analysis figures under %s", figures_dir)
