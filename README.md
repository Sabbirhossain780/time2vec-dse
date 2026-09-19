# Time2Vec-Enhanced Transformer for Sector-Wise Return Forecasting on the Dhaka Stock Exchange

A reproducible pipeline that forecasts next-day sector returns on the Dhaka
Stock Exchange with a SimpleRNN, an LSTM, and a small Time2Vec Transformer,
plus classical baselines and occlusion/integrated-gradients explainability.
Originally a set of Colab notebooks; now a scripted pipeline that rebuilds
every published number from the raw workbook.

**What this found:**

- The Transformer loses to the LSTM, to ARIMA, and often to "predict no
  change." That result is stable across 5 seeds.
- The reason is not tuning. Inside this architecture the attention block is
  measurably inert, and the layer named `Time2Vec` does not encode time.
- The Transformer's mean RMSE (1.091) is indistinguishable from predicting
  no change at all (1.092), and it beats that baseline in 12 of 25
  seed x sector cells — a coin flip.
- Five repair attempts — mean pooling, LR warmup/decay, a trainable
  positional embedding, removing Time2Vec, and implementing Time2Vec
  correctly — all failed. So did replacing self-attention with a fixed
  uniform average, which performs no differently from keeping it.
- Read [Limitations](#limitations) before citing anything here. This is one
  market, four years, and roughly ten model variants scored against a single
  test split. It is exploratory, not confirmatory.

## Method

Each sector's daily OHLCV series is smoothed with a 10-day moving average,
differenced into returns, and MinMax-scaled. An 8-day window of the 5 scaled
return features predicts the next day's `return_close`, many-to-one.
Predictions are inverted through the sector's scaler and added to the last
known close, so the reported RMSE/MAE are in price units.

Split is 80/10/10 chronological, no shuffling, with a 10-sample embargo at
each boundary (see [Protocol corrections](#protocol-corrections)). Training is
50 epochs of Adam with early stopping on validation loss (patience 5).
Every model is trained per-sector; the three core models are also trained
pooled across sectors (`OVERALL`).

Architectures: `SimpleRNN`; `LSTM`; and a Transformer encoder with a
Time2Vec embedding fused into a linear feature projection, one multi-head
self-attention block (`d_model=32`, 2 heads, `d_ff=64`, pre-LN), and a
take-last-position head. Baselines are naive persistence (predict zero
return) and ARIMA with an AIC grid search over `p,q ∈ 0..3`, `d ∈ {0,1}`
fitted on train+validation and walk-forward one-step forecast.

## Results

Every number below comes from a single sweep: 8 model variants × 5 sectors ×
5 seeds (42, 7, 123, 2024, 8675309), all trained back-to-back in one process
per seed so the RNG call sequence is identical across variants. RMSE is in
price units.

| Model | Mean RMSE | Run-to-run std | Beats naive |
|---|---|---|---|
| *Naive (predict no change)* | 1.092 | -- (deterministic) | -- |
| *ARIMA* | 0.812 | -- (deterministic) | 4/5 sectors |
| LSTM | 0.841 | 0.040 | 23/25 |
| RNN | 0.973 | 0.097 | 19/25 |
| Transformer | 1.091 | 0.273 | 12/25 |
| TransformerRealT2V | 1.143 | 0.188 | 11/25 |
| TransformerNoT2V | 1.152 | 0.241 | 13/25 |
| TransformerUniform | 1.182 | 0.281 | 11/25 |
| TransformerV2 | 1.223 | 0.196 | 9/25 |
| TransformerV3 | 1.239 | 0.213 | 12/25 |

| Sector | Naive | ARIMA | RNN | LSTM | Transformer |
|---|---|---|---|---|---|
| Engineering | 1.067 | 0.740 | 0.979 ± 0.066 (4/5) | 0.792 ± 0.034 (5/5) | 0.997 ± 0.162 (2/5) |
| Fuel & Power | 2.411 | 1.270 | 1.451 ± 0.089 (5/5) | 1.585 ± 0.109 (5/5) | 1.818 ± 0.253 (5/5) |
| IT Sector | 0.303 | 0.267 | 0.284 ± 0.012 (5/5) | 0.275 ± 0.003 (5/5) | 0.303 ± 0.009 (3/5) |
| Services & Real Estate | 0.449 | 0.640 | 1.050 ± 0.293 (0/5) | 0.432 ± 0.037 (3/5) | 0.755 ± 0.246 (0/5) |
| Telecommunication | 1.230 | 1.143 | 1.101 ± 0.027 (5/5) | 1.122 ± 0.018 (5/5) | 1.581 ± 0.695 (2/5) |

ARIMA has the best mean RMSE of anything here, and the LSTM is a close
second while beating the naive baseline in 23 of 25 cells. The Transformer
sits on top of the naive baseline: 1.091 against 1.092, winning 12 of 25
cells. Whatever it has learned is worth approximately nothing over assuming
tomorrow equals today.

Two sector-level results are worth not glossing over. In **Services & Real
Estate** the naive baseline beats everything, ARIMA included — the RNN and
Transformer lose in 5 of 5 seeds. In **Fuel & Power**, the sector with the
largest absolute errors, every model beats naive in 5 of 5 seeds, so the
picture is not uniform across sectors.

The Transformer's run-to-run standard deviation is 0.273 against the LSTM's
0.040 — roughly seven times larger. That is a reliability finding
independent of accuracy: same architecture, same data, a different seed, and
it lands somewhere materially different. Telecommunication is the worst case
at ±0.695 on a mean of 1.581.

## Why the Transformer underperforms

Four measurements, each independently verifiable from the code and a trained
model. Together they say the model is not a working Transformer that happens
to be badly tuned — two of its three distinguishing components do nothing.

**1. The `Time2Vec` layer does not encode time.** It applies `sin` and a
linear term element-wise to the five OHLCV *values* at each step
([models.py:15](src/aml/models.py:15)). Kazemi et al. define Time2Vec on the
time index τ. As implemented it is a per-feature nonlinear reparametrisation
that carries no information about which timestep it is. `TimeIndexTime2Vec`
([models.py:41](src/aml/models.py:41)) implements the paper's version for
comparison.

**2. The positional embedding never trains.** `tf.range` produces a concrete
eager tensor, so calling `layers.Embedding` on it executes eagerly and its
output is baked into the graph as a constant
([models.py:83](src/aml/models.py:83)):

```python
positions = tf.range(start=0, limit=seq_length, delta=1)
pos_emb = layers.Embedding(input_dim=seq_length, output_dim=d_model, name="pos_encoding")(positions)
```

`pos_encoding` is absent from `model.trainable_variables` — verified. The
model's positional signal is therefore frozen random noise at initialization,
measured at rms 0.029 against a content signal of rms 0.246, identical for
every sample in the batch. This is a general Keras Functional API trap: a
layer called on a non-symbolic tensor runs once and becomes a constant.

**3. Attention is near-uniform and query-independent.** On a trained
Engineering model over 85 test samples, attention weights span 0.109–0.144
against a uniform 0.125, vary negligibly with the query position, and put
their largest mass on `t-7` and `t-4` rather than on recent steps — which
contradicts what occlusion says the model actually relies on.

**4. Deleting the learned routing changes nothing.** `TransformerUniform`
replaces attention with a fixed uniform average over value projections,
dropping Q and K entirely ([models.py:130](src/aml/models.py:130)). It wins
14 of 25 paired cells against the real attention layer — a coin flip
(sign-test p ≈ 0.35) — while giving up 4,224 of the attention layer's 8,416
parameters. The learned routing is not buying anything measurable.

### Repair attempts

Each variant is paired against the original Transformer on identical
(seed, sector) cells from the same run. "Wins" counts cells where the
variant's RMSE is lower, out of 25.

| Variant | Mean RMSE | vs original | Wins (paired) |
|---|---|---|---|
| Transformer (original) | 1.091 | -- | -- |
| TransformerV2 | 1.223 | +0.132 | 7/25 |
| TransformerV3 | 1.239 | +0.148 | 10/25 |
| TransformerUniform | 1.182 | +0.091 | 14/25 |
| TransformerNoT2V | 1.152 | +0.061 | 8/25 |
| TransformerRealT2V | 1.143 | +0.053 | 9/25 |

**Not one variant beat the original on mean RMSE, and not one is
distinguishable from it by paired sign test** (14/25 is the best result, at
p ≈ 0.35). Read that as "none of these changed anything," not as a ranking.

- **`TransformerV2`** (mean pooling + linear warmup with cosine decay) cut
  run-to-run variance from 0.273 to 0.196 but has the second-worst mean.
  Mean pooling dilutes the recency signal take-last exploits, and occlusion
  shows this problem is heavily recency-dominated.
- **`TransformerV3`** (positional embedding built on a symbolic input so it
  actually trains) has the worst mean of any variant. **This disproves the
  obvious hypothesis** — which an earlier version of this README asserted —
  that the frozen positional embedding explains the underperformance. The
  frozen embedding is a genuine defect, and repairing it made things worse.
  Something else is binding.
- **`TransformerNoT2V`** and **`TransformerRealT2V`** respectively delete the
  Time2Vec branch and replace it with the paper's formulation. Both land
  within noise of the original, so the Time2Vec component is not where the
  performance is being lost either — removing it entirely costs nothing.
- **`TransformerUniform`** removes learned attention routing along with
  4,224 of the attention layer's 8,416 parameters, and performs no
  differently. Note the confound: it removes routing *and* parameters at
  once, so "attention is useless here" and "this model is overparameterised
  for ~670 training rows" are not separated by this ablation alone.

**A result that did not replicate.** Under the earlier, leaky protocol
`TransformerUniform` appeared to *improve* on the original (mean RMSE 1.167
to 1.106, beats-naive 9/25 to 15/25), and an earlier version of this README
reported that. Under the corrected protocol, with every variant re-run from
an identical fresh RNG state, it does not: the mean is worse and the paired
win rate is a coin flip. The weaker claim — that the learned routing
contributes nothing measurable — survives. The stronger one did not.

The honest summary is that attention is not contributing on an 8-day window
of a heavily smoothed target, and none of the standard fixes change that.
This is consistent with the broader finding that simple linear models are
competitive with transformer architectures on long-term forecasting
benchmarks ([Zeng et al., AAAI 2023](https://arxiv.org/abs/2205.13504)).

## What the XAI stage found

Occlusion sensitivity and integrated gradients over the three core models ×
5 sectors (`outputs/results/xai/xai_summary.csv`):

- **Occlusion: the most recent timestep `t-1` dominates in 14 of 15
  (model, sector) pairs.** Only Transformer/Telecommunication picks a
  different lag. Same story as the naive baseline from another angle — if
  yesterday's value does most of the work, "yesterday, unchanged" is hard to
  beat.
- **Integrated gradients agree for LSTM and RNN** (`t-1` in all 10 cases)
  but are scattered for the Transformer (`t-8, t-1, t-1, t-5, t-1`) —
  independent evidence that its learned representation is less stable.
- **No single OHLCV feature dominates.** `close`, `open`, `high` and `low`
  each top 3–4 of the 15 pairs; `volume` tops one. Nothing here justifies
  pruning an input feature.

XAI is a reimplementation — see [Known gaps](#known-gaps). It is computed
against seed-42 models and predates the protocol corrections below.

## Protocol corrections

Two look-ahead paths were found in the original preprocessing chain and
fixed. Every number in this repository's git history before this change came
from the leaky version.

The main ranking — ARIMA and LSTM ahead, Transformer level with naive —
survived the correction. One secondary finding did not: `TransformerUniform`
no longer improves on the original Transformer (see [Repair
attempts](#repair-attempts)). Treat pre-correction numbers as superseded
rather than as a comparable earlier measurement, since the model list and
RNG protocol changed at the same time.

1. **The scaler was fitted on the whole series** before splitting, letting
   validation and test extremes set the scale training data was normalized
   by. It is now fitted on training rows only
   ([preprocessing.py:38](src/aml/preprocessing.py:38)).
2. **The splits were contiguous with no embargo.** Targets are first
   differences of a 10-day moving average, so `y[i]` and `y[j]` share
   underlying days whenever `|i − j| < 10`; train and test labels overlapped
   across the boundary. A 10-sample embargo is now dropped at each boundary
   ([utils.py:17](src/aml/utils.py:17)). The test window itself is unchanged.

`tests/test_leakage.py` pins all of this down: sequences are causal, the
scaler never sees held-out rows, the embargo is wide enough to break the
label overlap, and anything aligned to the sample axis lines up with the
test block.

```bash
python tests/test_leakage.py
```

**Not fixed, deliberately:** the 10-day moving average uses
`min_periods=1`, so the first 9 rows of each sector are computed over partial
windows and are not comparable to the rest. This reproduces the original
notebook and affects training rows only; changing it would break the
byte-level reproduction of the archived run.

## Limitations

- **One market, four years, five sectors.** 2019 is absent from the raw
  workbook (its sheet uses a different header schema — reproduced faithfully
  and documented under [Data lineage](#data-lineage)). The window spans
  COVID. This is a case study, not evidence about market structure.
- **Exploratory, not confirmatory.** Roughly ten model variants have now been
  scored against the same test split, with no untouched holdout left. Under
  any multiple-testing correction, none of the comparisons here carry the
  nominal significance they would as a single pre-registered test. Treat
  effect directions as suggestive and effect sizes as unreliable.
- **No economic claim.** RMSE on a reconstructed price is not a tradable
  signal. There is no Sharpe ratio, no transaction cost, no position sizing.
- **Not a claim about Time2Vec.** The layer as implemented here is not
  Time2Vec. `TransformerRealT2V` tests the paper's formulation in this
  specific small-data setting only.
- **Five seeds is enough to separate a robust pattern from a lucky
  initialization**, which is what it is used for. It is not enough for a
  tight interval on any individual number.
- **Pooled OVERALL models are not reported here.** The `all` stage still
  trains them, but they were never seed-swept and their last published
  numbers came from the leaky protocol, so they are omitted rather than
  carried forward stale. XAI is likewise single-seed. ARIMA and the naive
  baseline are deterministic and need no sweep.
- Keras/TF training is not bit-deterministic across hardware even with a
  fixed seed.

## Known gaps

- **XAI is a fresh reimplementation, not a byte-for-byte reproduction.** No
  surviving notebook contains the code that produced
  `archive/reference_outputs/results/xai/*.csv` — only outputs and a summary
  table with permanently-empty IG/SHAP columns. `src/aml/xai.py` reimplements
  occlusion sensitivity (verified: reproduces the row/column-sum relationship
  between the archived `*_heat.csv` and `*_time.csv`/`*_feature.csv` files
  exactly) and integrated gradients from scratch. SHAP was never computed in
  the original run either, so it is not reproduced.
- **The `realworld` stage** ports the original notebook's Block 10, which was
  present but disabled. It is wired up and usable but untested against real
  unseen data, since none was ever supplied.
- Pooled-OVERALL real-world evaluation was dropped; only per-sector is
  implemented.

## Folder structure

```
data/raw/                   Raw DSE workbook -- source for stockprice.csv
data/processed/             stockprice.csv (gitignored; output of the prep-data stage)
src/aml/                    Pipeline package, one module per stage
run_pipeline.py             CLI
scripts/                    Analysis helpers (sweep -> README tables)
tests/                      Leakage and split-alignment guards
outputs/                    Figures, models, results (gitignored; output of the pipeline)
notebooks/archive/          Original Colab notebooks
docs/                       Experimental Setup & Results writeup (.docx)
archive/
  reference_outputs/        stockprice.csv + figures/models/results from the original notebook run
  Other/                    A duplicate rerun of the original notebook
  Transformer Model/        Duplicate of files already present in data/raw/ and notebooks/archive/
  legacy_2011_2015_lineage/ An earlier, unused data-prep lineage: comliled.ipynb, DSE-2011..2015.csv,
                            merged_excel.xlsx, top5/top10_companies.csv
```

### Data lineage

`stockprice.csv` (4270 rows, 2017-2021, 5 sectors) is reconstructed from
`data/raw/DSE-2017 to 2021.csv.xlsx` (sheets `DSE-2017`..`DSE-2021`), filtered
to 5 sectors, per the provenance note in `data/raw/DSE_master_filtered.xlsx`.
The reconstruction is verified against the archived original: 4270 rows, zero
mismatched cells.

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
recomputing it — `evaluate`/`visualize` load the saved run state rather than
retraining):

```bash
python run_pipeline.py prep-data                   # data/raw -> data/processed/stockprice.csv
python run_pipeline.py train --epochs 50           # per-sector + OVERALL models -> outputs/models
python run_pipeline.py evaluate                    # outputs/results/comparison_*.csv
python run_pipeline.py visualize                   # outputs/figures
python run_pipeline.py xai                         # outputs/results/xai
python run_pipeline.py baselines                   # naive + ARIMA
python run_pipeline.py realworld --input new.csv   # score saved models against fresh data
```

Reproducing the sweep that produced the tables above — one seed per process,
each writing durably, then merge and format:

```bash
MODELS="Transformer LSTM RNN TransformerV2 TransformerV3 TransformerUniform TransformerNoT2V TransformerRealT2V"
for S in 42 7 123 2024 8675309; do
    python run_pipeline.py multiseed --seed $S --models $MODELS --tag _purged
done
python run_pipeline.py multiseed-merge --tag _purged
python scripts/summarize_sweep.py --tag _purged
```

One seed per OS process is deliberate: each seed's results are written
immediately so a crash does not lose the others, and a long-lived process
training dozens of fresh Keras models back-to-back has been observed here to
hit a multi-minute XLA recompilation stall.

Useful flags: `--models` (subset; everything past `Transformer LSTM RNN` is
opt-in), `--seq-len` (lookback window; everything downstream adapts),
`--epochs`, `--batch-size`, `--strict-data-prep` (fail instead of warn if the
raw-data reconstruction misses the documented 4270 rows), `-v`.

### Checking a rerun reproduces the original results

```bash
diff archive/reference_outputs/results/comparison_overall.csv outputs/results/comparison_overall.csv
```

Exact numeric match is not guaranteed — Keras/TF ops are not bit-deterministic
across versions and hardware even with a fixed seed — but shapes and ballpark
metrics should agree. Note that the protocol corrections above intentionally
change the numbers relative to the archived run.

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


## License

[MIT](LICENSE).
