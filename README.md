# Time2Vec-Enhanced Transformer for Sector-Wise Return Forecasting on the Dhaka Stock Exchange

Predicts next-day return (and reconstructed closing price) per Dhaka Stock
Exchange sector using RNN, LSTM, and a small Time2Vec Transformer, trained
per-sector and pooled (OVERALL), with occlusion-sensitivity and
integrated-gradients explainability.

This was originally a set of Google Colab notebooks. It's now a scripted,
reproducible pipeline — see [Reproducing results](#reproducing-results). For
what the models actually found, see [Results & discussion](#results--discussion).

## Method

Each sector's daily OHLCV series is made (approximately) stationary via a
10-day moving-average smoothing pass, then differenced into returns, then
MinMax-scaled — a standard preprocessing chain for financial series that are
otherwise dominated by trend/level rather than the day-to-day dynamics the
models are meant to learn. An 8-day lookback window (`SEQ_LEN=8`) of the 5
scaled return features (open/high/low/close/volume) is used to predict the
next day's `return_close`, many-to-one. Predicted returns are inverted back
through the sector's fitted scaler and added to the last known close to get
an actual-price prediction, which is what all the reported RMSE/MAE figures
below are measured in (except the OVERALL table, which is reported in
normalized-return space — see the note on that table).

Three architectures are trained under identical preprocessing, sequence
length, and an 80/10/10 chronological train/val/test split (no shuffling
across time): a `SimpleRNN`, an `LSTM`, and a small Transformer encoder with
a learned [Time2Vec](https://arxiv.org/abs/1907.05321) time embedding fused
with a linear feature projection, one multi-head self-attention block, and a
"take the last position" head (`d_model=32`, 2 heads, `d_ff=64`). Each is
trained twice: once per-sector (5 independent models) and once pooled across
all sectors (`OVERALL`, ~14x more training rows). 50 epochs, Adam,
early-stopping on validation loss (patience 5), fixed seed — though see the
[reproducibility caveat](#reproducibility-caveat-read-before-citing-any-of-this)
before treating any single run's numbers as precise.

## Results & discussion

Numbers below are from the full 50-epoch GPU run in this repo's `outputs/`
(reproduce with `python run_pipeline.py all`); see
[Reproducing results](#reproducing-results) for how to regenerate them and
diff against the archived original.

### Pooled (OVERALL) comparison

Normalized-return space (MinMax-scaled returns, not actual price units —
these three numbers are only meaningfully comparable to each other, not to
the per-sector price-space table below):

| Model | Test MSE | Test MAE |
|---|---|---|
| **LSTM** | **0.00218** | **0.0239** |
| RNN | 0.00226 | 0.0236 |
| Transformer | 0.00243 | 0.0286 |

LSTM and RNN are close (RNN is actually marginally better on MAE); the
Transformer trails both by a consistent margin (+12% MSE, +20% MAE over
LSTM), despite having comparable or more trainable parameters and a strictly
richer architecture (attention + learned time embedding vs. a single
recurrent gate).

### Per-sector comparison, against a naive baseline and ARIMA -- checked across 5 seeds

The single most important sanity check missing from most retail stock-return
projects — including, to be clear, earlier versions of this one — is: does
the model beat "predict no change"? Does it beat a plain classical model?
And does the answer hold up, or was it one lucky weight initialization?
All three questions are answered here.

Two baselines, both using the exact same preprocessing, scaler, and
chronological test split as the trained models: a persistence baseline
(tomorrow's close = today's close), and a per-sector ARIMA model (order
selected by AIC grid search on train+val, evaluated with walk-forward
one-step forecasts through the test window -- `src/aml/baselines.py`,
`python run_pipeline.py baselines`). Both are deterministic given the data,
so they're reported as a single number. RNN/LSTM/Transformer are **not**
deterministic -- each was retrained from scratch under 5 different seeds
(42, 7, 123, 2024, 8675309; `src/aml/multiseed.py`, `python run_pipeline.py
multiseed --seed <n>` x5 then `multiseed-merge`), reported as mean ± std
RMSE across those 5 runs, with "beats naive" as a fraction (e.g. "5/5" means
every seed's model beat the naive baseline on that sector; "0/5" means none
did). All RMSE in actual price units on the test window.

> **Correction, for transparency:** an earlier version of this table
> normalized the naive baseline's RMSE and the trained models' RMSE by two
> different mean prices before converting to percentages (full-series vs.
> test-window), which flipped several verdicts -- most notably making IT
> Sector look artificially hard. Verdicts here are based on raw RMSE
> (same units, same test window for every model), which doesn't depend on
> any normalization choice.

| Sector | Naive | ARIMA | LSTM (mean±std, beats) | RNN (mean±std, beats) | Transformer (mean±std, beats) |
|---|---|---|---|---|---|
| Engineering | 1.067 | 0.740 (✓) | 0.81±0.04 (5/5) | 0.86±0.05 (5/5) | 1.29±0.30 (2/5) |
| Fuel & Power | 2.411 | 1.270 (✓) | 1.66±0.06 (5/5) | 1.55±0.22 (5/5) | 2.29±0.14 (4/5) |
| IT Sector | 0.303 | 0.267 (✓) | 0.27±0.00 (5/5) | 0.29±0.02 (4/5) | 0.29±0.01 (4/5) |
| Services & Real Estate | 0.449 | 0.640 (✗) | 0.47±0.08 (4/5) | 0.63±0.11 (0/5) | 0.88±0.27 (0/5) |
| Telecommunication | 1.230 | 1.143 (✓) | 1.12±0.01 (5/5) | 1.14±0.02 (5/5) | 1.40±0.11 (0/5) |

**What survives the robustness check, and what doesn't:**

- **LSTM's headline result softens but mostly holds.** It's robustly better
  than naive (5/5 seeds) in 4 of 5 sectors. In Services & Real Estate,
  though, it only beat naive in 4/5 seeds, and its *mean* RMSE (0.47) is
  actually slightly worse than naive (0.449) — the single-seed run that
  originally "confirmed" LSTM beats naive everywhere was, on this one
  sector, closer to a coin flip than a real edge.
- **RNN is a genuine mixed bag, and consistently so.** Robust (5/5) in the
  same 3 sectors as LSTM (Engineering, Fuel & Power, Telecommunication), but
  it lost to naive in **every single one** of the 5 seeds in Services &
  Real Estate (0/5) — not noise, a consistent failure.
- **The Transformer is both the weakest model and by far the least
  reproducible one.** Averaged across sectors, its run-to-run standard
  deviation (0.165 RMSE units) is **~4x LSTM's (0.040) and ~2x RNN's
  (0.084)**. That instability is itself a finding, separate from accuracy:
  in Engineering, the original single-seed run showed the Transformer
  clearly losing to naive — but across 5 seeds it actually *beats* naive
  40% of the time, meaning that single run wasn't even a reliable
  description of the Transformer's own typical behavior, let alone a fair
  comparison to the other models. It never beat naive, in any of the 5
  seeds, in Services & Real Estate or Telecommunication.
- **Services & Real Estate is now solidly confirmed as the hard sector for
  everyone** — RNN and the Transformer lose to naive in all 5 seeds each,
  ARIMA loses outright, and even LSTM's edge there is marginal at best.
  This is a much stronger claim than the single-run version could support.
- **ARIMA remains competitive** — best single model in 3/5 sectors, still
  with zero training and no variance to worry about, though it's now being
  compared against *distributions*, not point estimates, for the neural
  models.

The practical lesson, independent of any specific number: **a single
training run of a small neural network on a small per-sector financial
dataset is not a reliable basis for a "beats baseline" claim.** Several of
this project's own earlier verdicts (in an earlier version of this
README) would not have survived this check.

### Trying to fix the Transformer: a targeted experiment (mixed result)

Two specific, targeted fixes for the instability/underperformance diagnosed
above: (1) mean pooling over all 8 timesteps instead of reading out only the
last position, and (2) a linear warmup + cosine-decay learning rate schedule
instead of flat Adam -- both known remedies for exactly the kind of
optimization instability the multi-seed sweep exposed. Implemented as
`TransformerV2` (`src/aml/models.py`, `build_transformer_model_v2`) and
compared against the original `Transformer` retrained under **identical
conditions** (same 5 seeds, same run, trained back-to-back per sector, so
the comparison isn't confounded by a different RNG call sequence across
separate runs -- reproduce with `python run_pipeline.py multiseed --seed
<n> --models Transformer TransformerV2 --tag _v1v2`, x5, then
`multiseed-merge --tag _v1v2`).

| | Transformer (original) | TransformerV2 |
|---|---|---|
| Mean RMSE, avg across sectors | **1.055** | 1.119 |
| Std (run-to-run stability), avg across sectors | 0.197 | **0.124** (−37%) |
| Beats naive, all 25 seed×sector combinations | **12/25 (48%)** | 8/25 (32%) |

**It's a real, mixed result, not a clean win.** TransformerV2 is
meaningfully more stable (lower run-to-run variance in 4/5 sectors,
sometimes dramatically -- Services & Real Estate's std fell from 0.332 to
0.095), confirming the training-instability diagnosis was correct. But mean
accuracy and the beats-naive rate both got *worse* overall, including a
real regression in Fuel & Power (RMSE 1.85 -> 2.21; beats-naive dropped
from 5/5 to 3/5 seeds -- previously the one sector the original Transformer
was actually reliable in).

**Best guess why:** mean pooling assumes every timestep deserves roughly
equal weight in the final prediction. But the XAI results above already
show this problem is heavily recency-biased -- `t-1` dominates occlusion
sensitivity in 14/15 (model, sector) combinations. Take-last, for all its
"discards information" framing, was actually a reasonable architectural
match for a recency-dominated problem: it reads out exactly the position
that matters most. Mean pooling dilutes that signal by averaging in 7
comparatively uninformative earlier timesteps. The fix addressed the
instability correctly, but at the cost of fighting the data's own
structure -- which take-last happened to exploit, likely by accident
rather than design.

**Left as a follow-up, not implemented:** an attention-weighted pooling
(learn the weights instead of hand-picking mean or last), or the warmup
schedule alone without the pooling change, to isolate which of the two
fixes is actually responsible for the stability gain.

### What the XAI stage found

`xai_summary.csv` (occlusion sensitivity + integrated gradients, all 3
models × 5 sectors — see [Known gaps](#known-gaps--honest-caveats) for why
this is a reimplementation, not a reproduction of the original numbers):

- **Occlusion: the most recent timestep (`t-1`) dominates in 14 of 15
  (model, sector) combinations.** Only Transformer/Telecommunication picked
  a different lag (`t-6`). This is the same story as the naive-baseline
  result from a different angle: if yesterday's value is doing nearly all
  the work the model uses, it's unsurprising that "yesterday's value,
  unchanged" is a hard baseline to beat.
- **Integrated Gradients tells a consistent story for LSTM/RNN (`t-1` in
  all 10 cases) but a noisier one for the Transformer** (`t-8`, `t-1`,
  `t-1`, `t-5`, `t-1` across its 5 sectors) — mild independent evidence, on
  top of the accuracy numbers above, that the Transformer's learned
  representation is less stable here, not just less accurate.
- **No single OHLCV feature dominates** — `close`, `open`, `high`, and `low`
  each show up as the top occlusion feature in 3-4 of the 15 combinations;
  `volume` shows up once. There isn't an obviously redundant input feature
  to prune based on this evidence.

### Reproducibility caveat (read before citing any of this)

The **per-sector RNN/LSTM/Transformer comparison is now checked across 5
seeds** (see above) -- that part of this README is no longer a single-run
claim. What's still single-run / not seed-swept:

- The **pooled OVERALL comparison** (normalized-return MSE/MAE table,
  above) -- only one seed per model.
- **ARIMA and the naive baseline** -- deterministic given the data, so
  there's no seed to vary; their only source of run-to-run variation would
  be a different train/test split or different raw data.
- The **XAI results** (occlusion sensitivity, integrated gradients) --
  computed once, against the seed-42 models specifically.

Keras/TF training is also not bit-deterministic across hardware/backends
even with a fixed seed (see [Checking a rerun reproduces the original
results](#checking-a-rerun-reproduces-the-original-results) — a
from-scratch GPU rerun of the full pipeline landed within ~5-15% of the
archived original run on every headline metric). 5 seeds is enough to
distinguish "robust pattern" from "one-run fluke" (which is what it was
used for above) but is not enough for a tight confidence interval on any
individual number -- a paper-grade claim would want more seeds, and would
also sweep the pooled OVERALL models and re-run XAI per seed rather than
against one arbitrarily-chosen model instance.

## Folder structure

```
data/raw/                  Raw DSE workbook -- source for stockprice.csv
data/processed/             stockprice.csv (gitignored; output of the prep-data stage)
src/aml/                    Pipeline package, one module per stage
run_pipeline.py             CLI
outputs/                    Figures, models, results (gitignored; output of the pipeline)
notebooks/archive/           Original Colab notebooks
docs/                        Experimental Setup & Results writeup (.docx)
archive/
  reference_outputs/         stockprice.csv + figures/models/results from the original notebook run
  Other/                     A duplicate rerun of the original notebook
  Transformer Model/         Duplicate of files already present in data/raw/ and notebooks/archive/
  legacy_2011_2015_lineage/  An earlier, unused data-prep lineage: comliled.ipynb, DSE-2011..2015.csv,
                              merged_excel.xlsx, top5/top10_companies.csv
```

### Data lineage

`stockprice.csv` (4270 rows, 2017-2021, 5 sectors) is reconstructed from
`data/raw/DSE-2017 to 2021.csv.xlsx` (sheets `DSE-2017`..`DSE-2021`), filtered
to 5 sectors, per the provenance note in `data/raw/DSE_master_filtered.xlsx`.

`archive/legacy_2011_2015_lineage/` is a separate, unused lineage: an earlier
experiment on 2011-2015 data pulled from a since-deleted external folder
(`D:\AML Project\real world data`). It is not wired into the pipeline.

## Reproducing results

```bash
pip install -r requirements.txt

# Full pipeline: raw data -> stockprice.csv -> train -> evaluate -> figures -> XAI
python run_pipeline.py all

# Fast wiring check (1 epoch, skips XAI) before committing to a full run
python run_pipeline.py all --smoke-test
```

Individual stages (each reuses prior stage output where possible instead of
recomputing it -- e.g. `evaluate`/`visualize` load the saved training run
state rather than retraining):

```bash
python run_pipeline.py prep-data              # data/raw -> data/processed/stockprice.csv
python run_pipeline.py train --epochs 50       # per-sector + OVERALL models -> outputs/models
python run_pipeline.py evaluate                # outputs/results/comparison_*.csv
python run_pipeline.py visualize               # outputs/figures
python run_pipeline.py xai                     # outputs/results/xai
python run_pipeline.py baselines               # outputs/results/baselines_comparison.csv (naive + ARIMA)
python run_pipeline.py realworld --input new.csv   # score saved models against fresh data

# Multi-seed robustness check: one seed per invocation (each saves durably), then merge
python run_pipeline.py multiseed --seed 42   # ...repeat for 7, 123, 2024, 8675309 (or your own seeds)
python run_pipeline.py multiseed-merge       # outputs/results/multiseed_comparison.csv + _summary.csv

# TransformerV2 experiment: paired with the original under identical seeds (--tag keeps
# it from overwriting the baseline multiseed files above)
python run_pipeline.py multiseed --seed 42 --models Transformer TransformerV2 --tag _v1v2
python run_pipeline.py multiseed-merge --tag _v1v2
```

Useful flags: `--models Transformer LSTM` (subset; `TransformerV2` is opt-in,
not part of the default set), `--epochs`, `--batch-size`, `--strict-data-prep`
(fail instead of warn if the raw-data reconstruction doesn't match the
documented 4270-row provenance), `-v` for debug logging.

### Checking a rerun reproduces the original results

```bash
diff archive/reference_outputs/results/comparison_overall.csv outputs/results/comparison_overall.csv
```

Exact numeric match isn't guaranteed (Keras/TF ops aren't bit-deterministic
across versions/hardware even with a fixed seed), but shape and ballpark
metrics should agree closely.

## GPU training (WSL2)

`pip install tensorflow` on native Windows has been CPU-only since TF 2.11
(Google's own decision, not a local misconfiguration). To use an NVIDIA GPU on
Windows, run this project inside WSL2 instead:

```bash
# One-time setup, from a Windows shell (skip if WSL2 + a distro already exists):
wsl --install

# Inside the WSL distro (e.g. Ubuntu):
curl -LsSf https://astral.sh/uv/install.sh | sh   # prebuilt Python binaries, no compiling
source $HOME/.local/bin/env
uv python install 3.12                            # match a Python version TF actually ships wheels for
uv venv ~/venvs/aml-gpu --python 3.12
cd /mnt/<drive>/AML                                 # this project, via the Windows-drive mount
uv pip install --python ~/venvs/aml-gpu -r requirements.txt "tensorflow[and-cuda]"
```

`tensorflow[and-cuda]` pulls matching CUDA/cuDNN as regular pip packages, so
no manual system-wide CUDA toolkit install is needed inside WSL. It relies on
the Windows NVIDIA driver via WSL2 GPU passthrough (`/usr/lib/wsl/lib`),
which is already there if `nvidia-smi` works on the Windows side.

**Known papercut on some distro/glibc combinations:** TensorFlow's RPATH-based
auto-discovery of the pip-installed CUDA libraries can fail to register the
GPU (`tf.config.list_physical_devices('GPU')` returns `[]`) even though
everything installed correctly. Fix: point `LD_LIBRARY_PATH` at the installed
`nvidia-*` package lib dirs. Bake it into the venv so it's automatic on every
`source ~/venvs/aml-gpu/bin/activate`:

```bash
V=~/venvs/aml-gpu
LIBDIRS=$(find "$V/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')
echo "export LD_LIBRARY_PATH=\"${LIBDIRS}\${LD_LIBRARY_PATH:-}\"" >> "$V/bin/activate"
```

Then run the pipeline as usual, from inside the activated venv:

```bash
source ~/venvs/aml-gpu/bin/activate
cd /mnt/<drive>/AML
python run_pipeline.py all --smoke-test   # verify: look for "Created device .../GPU:0" in the log
```

## Known gaps / honest caveats

- **XAI is a fresh reimplementation, not a byte-for-byte reproduction.** No
  notebook here ever contained the code that produced
  `archive/reference_outputs/results/xai/*.csv` — only the output files and
  a summary table (with permanently-empty IG/SHAP columns) survived, plus one
  orphaned partial Integrated-Gradients file. `src/aml/xai.py` reimplements
  occlusion sensitivity (verified: reproduces the row/column-sum relationship
  between the archived `*_heat.csv` and `*_time.csv`/`*_feature.csv` files
  exactly) and Integrated Gradients from scratch. SHAP was never actually
  computed in the original run either, so it's not reproduced.
- **Real-world evaluation (`realworld` stage)** ports the original notebook's
  Block 10, which existed but was disabled (commented out, no real input file
  was ever set). It's wired up and usable, but untested against real unseen
  data since none was ever supplied.
- The pooled-OVERALL real-world evaluation (present in the original Block 10)
  was dropped for simplicity; only per-sector real-world evaluation is
  implemented.

## License

[MIT](LICENSE).
