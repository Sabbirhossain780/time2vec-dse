"""Turn the merged multi-seed sweep into the markdown tables used in the README.

Usage: python scripts/summarize_sweep.py [--tag _purged]
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


def load(tag: str):
    rd = get_paths().results_dir
    summary = pd.read_csv(rd / f"multiseed_summary{tag}.csv")
    sweep = pd.read_csv(rd / f"multiseed_comparison{tag}.csv")
    return summary, sweep, rd


def headline(summary: pd.DataFrame) -> str:
    rows = []
    for m in CORE + VARIANTS:
        sub = summary[summary["Model"] == m]
        if sub.empty:
            continue
        beats = int(round((sub["Frac_seeds_beat_naive"] * sub["n_seeds"]).sum()))
        total = int(sub["n_seeds"].sum())
        rows.append((m, sub["Mean_RMSE"].mean(), sub["Std_RMSE"].mean(), beats, total))
    rows.sort(key=lambda r: r[1])
    out = ["| Model | Mean RMSE | Run-to-run std | Beats naive |", "|---|---|---|---|"]
    for m, mean, std, beats, total in rows:
        out.append(f"| {m} | {mean:.3f} | {std:.3f} | {beats}/{total} |")
    return "\n".join(out)


def per_sector(summary: pd.DataFrame) -> str:
    out = ["| Sector | Naive | RNN | LSTM | Transformer |", "|---|---|---|---|---|"]
    for sector in sorted(summary["Sector"].unique()):
        sub = summary[summary["Sector"] == sector]
        naive = sub["Naive_RMSE"].iloc[0]
        cells = [sector, f"{naive:.3f}"]
        for m in CORE:
            r = sub[sub["Model"] == m]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            n_beat = int(round(r["Frac_seeds_beat_naive"] * r["n_seeds"]))
            cells.append(f"{r['Mean_RMSE']:.3f} ± {r['Std_RMSE']:.3f} ({n_beat}/{int(r['n_seeds'])})")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def paired(sweep: pd.DataFrame) -> str:
    """Each variant vs the original Transformer on identical (seed, sector)."""
    key = ["Seed", "Sector"]
    base = sweep[sweep["Model"] == "Transformer"].set_index(key)["Test_RMSE_Actual_Return"]
    out = ["| Variant | Mean RMSE | vs original | Wins (paired) |", "|---|---|---|---|"]
    out.append(f"| Transformer (original) | {base.mean():.3f} | -- | -- |")
    for m in VARIANTS:
        sub = sweep[sweep["Model"] == m].set_index(key)["Test_RMSE_Actual_Return"]
        if sub.empty:
            continue
        joined = pd.concat([base.rename("base"), sub.rename("var")], axis=1).dropna()
        wins = int((joined["var"] < joined["base"]).sum())
        delta = joined["var"].mean() - joined["base"].mean()
        out.append(f"| {m} | {sub.mean():.3f} | {delta:+.3f} | {wins}/{len(joined)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_purged")
    args = ap.parse_args()

    summary, sweep, rd = load(args.tag)
    text = "\n\n".join([
        "## Headline (avg across 5 sectors, 5 seeds)", headline(summary),
        "## Per-sector: mean RMSE ± std (seeds beating naive)", per_sector(summary),
        "## Transformer variants, paired against the original", paired(sweep),
    ])
    print(text)
    (rd / f"summary_tables{args.tag}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
