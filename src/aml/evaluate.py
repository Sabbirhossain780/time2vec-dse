"""Stage 3: build the per-sector and OVERALL comparison tables and save them as CSVs.

Mirrors the original notebook's Block 7 -- minus the defensive re-training
fallback that existed only to survive out-of-order Colab cell execution.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _last_or_nan(x):
    return float(x[-1]) if isinstance(x, (list, tuple)) and len(x) else (float(x) if np.ndim(x) == 0 else np.nan)


def _hist_to_row(hist, model_type, sector):
    h = hist.history
    return {
        "Model": model_type,
        "Sector": sector,
        "Train_MSE_last": _last_or_nan(h.get("loss", [])),
        "Val_MSE_last": _last_or_nan(h.get("val_loss", [])),
        "Train_MAE_last": _last_or_nan(h.get("mae", [])),
        "Val_MAE_last": _last_or_nan(h.get("val_mae", [])),
    }


def build_comparison_tables(sectors, model_types, model_histories, all_results,
                             model_histories_overall, all_results_overall, results_dir: Path):
    results_dir = Path(results_dir)

    rows = []
    for model_type in model_types:
        for sector in sectors:
            hist = model_histories.get(model_type, {}).get(sector)
            if hist is None:
                continue
            row = _hist_to_row(hist, model_type, sector)
            res = all_results.get(model_type, {}).get(sector, {})
            row["Test_RMSE_Actual_Return"] = res.get("RMSE_Actual_Return", np.nan)
            row["Test_MAE_Actual_Return"] = res.get("MAE_Actual_Return", np.nan)
            rows.append(row)
    df_per_sector = pd.DataFrame(rows).sort_values(["Model", "Sector"])

    rows_overall = []
    for model_type, hist in model_histories_overall.items():
        row = _hist_to_row(hist, model_type, "OVERALL")
        test = all_results_overall.get(model_type, {})
        row["Test_MSE"] = test.get("Test_MSE", np.nan)
        row["Test_MAE"] = test.get("Test_MAE", np.nan)
        rows_overall.append(row)
    df_overall = pd.DataFrame(rows_overall).sort_values(["Model"])

    per_sector_csv = results_dir / "comparison_per_sector.csv"
    overall_csv = results_dir / "comparison_overall.csv"
    df_per_sector.to_csv(per_sector_csv, index=False)
    df_overall.to_csv(overall_csv, index=False)
    logger.info("Saved: %s", per_sector_csv)
    logger.info("Saved: %s", overall_csv)

    return df_per_sector, df_overall
