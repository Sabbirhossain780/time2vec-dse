"""Stage 0: build the modeling-ready stockprice.csv from the raw DSE workbook.

Source: data/raw/DSE-2017 to 2021.csv.xlsx, sheets DSE-2017..DSE-2021.
Filtered to the 5 sectors documented in data/raw/DSE_master_filtered.xlsx's
provenance note, which also records the expected row count (4270) used here
to validate the reconstruction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import EXPECTED_ROW_COUNT, RAW_SECTOR_CODE_MAP

logger = logging.getLogger(__name__)

RAW_SHEETS = ["DSE-2017", "DSE-2018", "DSE-2019", "DSE-2020", "DSE-2021"]


def _load_raw_sheets(raw_xlsx: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(raw_xlsx)
    frames = []
    for sheet in RAW_SHEETS:
        sheet_df = xl.parse(sheet)
        # Normalize column names per-sheet, before concatenation: the DSE-2019
        # sheet header uses "Open" instead of "Open " (trailing space) used by
        # the others. Cleaning per-sheet first means both map to the same
        # "open" column on concat, instead of surviving as two raw-distinct
        # columns ("Open " vs "Open") that would only collide into duplicate
        # "open" columns after a later global rename -- corrupting the output.
        sheet_df.columns = [c.strip().lower() for c in sheet_df.columns]
        frames.append(sheet_df)
    return pd.concat(frames, ignore_index=True)


OUTPUT_COLUMNS = ["sector", "trading_date", "open", "high", "low", "close", "volume", "year", "month", "date"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"trading date": "trading_date"})

    # NOTE: the DSE-2019 sheet uses different headers ("Company Name" instead
    # of "Sector"), so it has no "sector" column at all. Those rows fall out
    # here (sector is NaN -> not in RAW_SECTOR_CODE_MAP), which reproduces the
    # original pipeline's output exactly (verified: 2017/2018/2020/2021 only,
    # no 2019, matching archive/reference_outputs/stockprice.csv row-for-row).
    # This is a latent data-quality issue in the raw workbook, not a bug here.
    df = df[df["sector"].isin(RAW_SECTOR_CODE_MAP.keys())].copy()
    df["sector"] = df["sector"].map(RAW_SECTOR_CODE_MAP)

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df = df.dropna(subset=["trading_date"])

    df["year"] = df["trading_date"].dt.year
    df["month"] = df["trading_date"].dt.month
    df["date"] = df["trading_date"].dt.day

    df = df.sort_values(["sector", "trading_date"]).reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


def build_stockprice_csv(raw_xlsx: Path, output_csv: Path, strict: bool = False) -> pd.DataFrame:
    """Reconstruct stockprice.csv from the raw DSE workbook.

    If strict, raises when the row count doesn't match the documented
    provenance (4270); otherwise logs a warning so the pipeline still runs
    against updated raw data.
    """
    logger.info("Reading raw workbook: %s", raw_xlsx)
    raw = _load_raw_sheets(Path(raw_xlsx))
    df = _clean(raw)

    if len(df) != EXPECTED_ROW_COUNT:
        msg = (f"Row count mismatch: got {len(df)}, expected {EXPECTED_ROW_COUNT} "
               "(per data/raw/DSE_master_filtered.xlsx provenance note).")
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows -> %s", len(df), output_csv)
    return df
