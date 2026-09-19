"""Turn the multi-seed sweep into the markdown tables used in the README.

Reads the per-(seed, model) result files written by one process each
(`outputs/results/multiseed_seed_<seed><tag>_<model>.csv`) plus the naive and
ARIMA rows from `baselines_comparison.csv`, and writes
`outputs/results/summary_tables<tag>.md`.

Usage:
    python run_pipeline.py baselines            # refresh naive + ARIMA first
    python scripts/summarize_sweep.py --tag _purged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aml.config import get_paths  # noqa: E402

CORE = ["RNN", "LSTM", "Transformer"]
VARIANTS = ["TransformerV2", "TransformerV3", "TransformerUniform",
            "TransformerNoT2V", "TransformerRealT2V"]
RMSE = "Test_RMSE_Actual_Return"


def load_sweep(results_dir: Path, tag: str) -> pd.DataFrame:
    files = sorted(results_dir.glob(f"multiseed_seed_*{tag}_*.csv"))
    if not files:
        raise SystemExit(f"no sweep files matching multiseed_seed_*{tag}_*.csv in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dupes = df.duplicated(subset=["Seed", "Model", "Sector"]).sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicated (seed, model, sector) rows -- stale files present")
    return df


def load_baselines(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "baselines_comparison.csv"
    if not path.exists():
        raise SystemExit("baselines_comparison.csv missing -- run `python run_pipeline.py baselines`")
    df = pd.read_csv(path)
    return df[df["Model"].isin(["Naive", "ARIMA"])][["Model", "Sector", RMSE]]


def headline(sweep: pd.DataFrame, base: pd.DataFrame) -> str:
    naive = base[base["Model"] == "Naive"].set_index("Sector")[RMSE]
    rows = []
    for m in CORE + VARIANTS:
        sub = sweep[sweep["Model"] == m]
        if sub.empty:
            continue
        per_sector = sub.groupby("Sector")[RMSE]
        beats = int((sub[RMSE] < sub["Sector"].map(naive)).sum())
        rows.append((m, per_sector.mean().mean(), per_sector.std().mean(), beats, len(sub)))
    rows.sort(key=lambda r: r[1])

    arima = base[base["Model"] == "ARIMA"].set_index("Sector")[RMSE]
    out = ["| Model | Mean RMSE | Run-to-run std | Beats naive |", "|---|---|---|---|"]
    out.append(f"| *Naive (predict no change)* | {naive.mean():.3f} | -- (deterministic) | -- |")
    out.append(f"| *ARIMA* | {arima.mean():.3f} | -- (deterministic) | "
               f"{int((arima < naive).sum())}/{len(arima)} sectors |")
    for m, mean, std, beats, n in rows:
        out.append(f"| {m} | {mean:.3f} | {std:.3f} | {beats}/{n} |")
    return "\n".join(out)


def per_sector(sweep: pd.DataFrame, base: pd.DataFrame) -> str:
    naive = base[base["Model"] == "Naive"].set_index("Sector")[RMSE]
    arima = base[base["Model"] == "ARIMA"].set_index("Sector")[RMSE]

    out = ["| Sector | Naive | ARIMA | RNN | LSTM | Transformer |",
           "|---|---|---|---|---|---|"]
    for sector in sorted(sweep["Sector"].unique()):
        cells = [sector, f"{naive.get(sector, float('nan')):.3f}",
                 f"{arima.get(sector, float('nan')):.3f}"]
        for m in CORE:
            sub = sweep[(sweep["Model"] == m) & (sweep["Sector"] == sector)][RMSE]
            if sub.empty:
                cells.append("--")
                continue
            n_beat = int((sub < naive.get(sector, float("inf"))).sum())
            cells.append(f"{sub.mean():.3f} ± {sub.std():.3f} ({n_beat}/{len(sub)})")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def paired(sweep: pd.DataFrame) -> str:
    """Each variant against the original Transformer on identical (seed, sector)."""
    key = ["Seed", "Sector"]
    ref = sweep[sweep["Model"] == "Transformer"].set_index(key)[RMSE]
    out = ["| Variant | Mean RMSE | vs original | Wins (paired) |", "|---|---|---|---|"]
    out.append(f"| Transformer (original) | {ref.mean():.3f} | -- | -- |")
    for m in VARIANTS:
        sub = sweep[sweep["Model"] == m].set_index(key)[RMSE]
        if sub.empty:
            continue
        j = pd.concat([ref.rename("ref"), sub.rename("var")], axis=1).dropna()
        wins = int((j["var"] < j["ref"]).sum())
        out.append(f"| {m} | {sub.mean():.3f} | {j['var'].mean() - j['ref'].mean():+.3f} "
                   f"| {wins}/{len(j)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_purged")
    args = ap.parse_args()

    rd = get_paths().results_dir
    sweep = load_sweep(rd, args.tag)
    base = load_baselines(rd)

    n_seeds = sweep["Seed"].nunique()
    print(f"# {len(sweep)} rows | {sweep['Model'].nunique()} models "
          f"| {sweep['Sector'].nunique()} sectors | {n_seeds} seeds\n")

    text = "\n\n".join([
        headline(sweep, base),
        per_sector(sweep, base),
        paired(sweep),
    ])
    print(text)
    (rd / f"summary_tables{args.tag}.md").write_text(text, encoding="utf-8")
    sweep.to_csv(rd / f"multiseed_comparison{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
