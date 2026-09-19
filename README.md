# Time2Vec-Enhanced Transformer for Sector-Wise Return Forecasting on the Dhaka Stock Exchange

This repository has two parts.

**Part 1** is the project as it was planned: take five sectors of the Dhaka
Stock Exchange, and try to predict the next day's return with three neural
networks — a SimpleRNN, an LSTM, and a small Transformer using a Time2Vec
time embedding. Compare them against each other and against classical
baselines.

**Part 2** is what happened when the Transformer didn't work. Rather than
keep tuning it, we took it apart and measured what its components were
actually doing. Most of them were doing nothing, for reasons that don't show
up in any accuracy table.

Both parts are here, with the code to reproduce them.

---

# Part 1 — Forecasting DSE sector returns

## The data and the setup

Daily open/high/low/close/volume figures for five DSE sectors — Engineering,
Fuel & Power, IT, Services & Real Estate, and Telecommunication — covering
2017 to 2021. After cleaning, 4,270 rows, rebuilt from the raw workbook by
the pipeline in this repo (`data/raw/DSE-2017 to 2021.csv.xlsx`). 2019 is
missing, because that year's sheet in the source workbook uses a different
column layout; this matches the original study and is documented under
[Data lineage](#data-lineage).

Each sector's prices are smoothed with a 10-day moving average, then
differenced into returns, then scaled. The model sees 8 days of those five
return features and predicts the next day's close return. The prediction is
converted back to a price by adding it to the last known close, so all the
error figures below are in the same units as the share prices.

Data is split 80/10/10 in time order — train on the earliest stretch,
validate on the middle, test on the most recent. No shuffling, because
shuffling a time series lets the model see the future. That leaves roughly
670 training rows per sector, which is a small dataset by any standard.

## The models

- **SimpleRNN** and **LSTM** — standard recurrent networks.
- **Transformer** — a small encoder: a Time2Vec embedding added to a linear
  projection of the features, one multi-head self-attention block
  (`d_model=32`, 2 heads), and a head that reads the last position.
- **Naive baseline** — predict no change. Tomorrow equals today.
- **ARIMA** — a classical statistical model, order chosen by AIC search,
  refit walk-forward one step at a time.

The naive baseline matters more than it sounds. A lot of stock-prediction
results look impressive until you check whether they beat "assume nothing
changes," and many don't.

Every model was trained five times with different random starting weights
(seeds 42, 7, 123, 2024, 8675309), across all five sectors. That's 25
independent results per model, so a single lucky run can't carry a
conclusion.

## Results

Lower RMSE is better. "Beats naive" counts how many of the 25 seed × sector
runs came in under the naive baseline.

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

Per sector, for the three original models:

| Sector | Naive | ARIMA | RNN | LSTM | Transformer |
|---|---|---|---|---|---|
| Engineering | 1.067 | 0.740 | 0.979 ± 0.066 (4/5) | 0.792 ± 0.034 (5/5) | 0.997 ± 0.162 (2/5) |
| Fuel & Power | 2.411 | 1.270 | 1.451 ± 0.089 (5/5) | 1.585 ± 0.109 (5/5) | 1.818 ± 0.253 (5/5) |
| IT Sector | 0.303 | 0.267 | 0.284 ± 0.012 (5/5) | 0.275 ± 0.003 (5/5) | 0.303 ± 0.009 (3/5) |
| Services & Real Estate | 0.449 | 0.640 | 1.050 ± 0.293 (0/5) | 0.432 ± 0.037 (3/5) | 0.755 ± 0.246 (0/5) |
| Telecommunication | 1.230 | 1.143 | 1.101 ± 0.027 (5/5) | 1.122 ± 0.018 (5/5) | 1.581 ± 0.695 (2/5) |

**What this says, plainly:**

ARIMA — the oldest and simplest method here — has the lowest average error.
The LSTM is close behind and is the most dependable of the neural models,
beating the naive baseline in 23 of 25 runs.

The Transformer does not work. Its average error is 1.091; the naive
baseline's is 1.092. It beats "assume nothing changes" in 12 of 25 runs,
which is a coin flip. It is also unstable: retrain it with a different
random seed and the error moves by ±0.273 on average, about seven times the
LSTM's ±0.040. In Telecommunication it swings by ±0.695 on a mean of 1.581.

Two sectors are worth noting. In **Services & Real Estate**, nothing beats
the naive baseline reliably — not even ARIMA. In **Fuel & Power**, which has
the largest errors in absolute terms, every model beats naive in all five
runs. The picture isn't uniform.

---

# Part 2 — Why the Transformer didn't work

## First we tried to fix it

The obvious explanations each got a targeted fix, and each fix got trained
under the same five seeds and compared against the original on identical
runs:

| Variant | What changed | Mean RMSE | vs original | Wins (paired) |
|---|---|---|---|---|
| Transformer (original) | — | 1.091 | -- | -- |
| TransformerV2 | mean pooling + LR warmup/decay | 1.223 | +0.132 | 7/25 |
| TransformerV3 | positional embedding that actually trains | 1.239 | +0.148 | 10/25 |
| TransformerUniform | self-attention replaced by a fixed average | 1.182 | +0.091 | 14/25 |
| TransformerNoT2V | Time2Vec removed entirely | 1.152 | +0.061 | 8/25 |
| TransformerRealT2V | Time2Vec as the paper actually defines it | 1.143 | +0.053 | 9/25 |

None of them helped. Every one has a worse average than the original, and
none is far enough from it to be distinguishable from chance.

At that point, guessing at fixes clearly wasn't going to get anywhere. So
instead of changing the model, we measured it.

## What the measurements found

**The Time2Vec layer doesn't encode time.** Time2Vec, as defined in the
[original paper](https://arxiv.org/abs/1907.05321), applies a sine and a
linear term to the *time index* — to "which step is this." The layer in this
model ([models.py:15](src/aml/models.py:15)) applies them to the five OHLCV
*values* instead. It's a reshuffling of the numbers that carries no
information about position in the sequence at all. It shares a name with
Time2Vec and not much else.

**The positional embedding never trains.** Here is the line
([models.py:83](src/aml/models.py:83)):

```python
positions = tf.range(start=0, limit=seq_length, delta=1)
pos_emb = layers.Embedding(input_dim=seq_length, output_dim=d_model, name="pos_encoding")(positions)
```

`tf.range` produces an ordinary tensor with real values in it, not a
placeholder for future input. So the `Embedding` layer runs immediately,
once, and its random starting output gets frozen into the model as a fixed
constant. It never becomes a trainable weight. You can check: `pos_encoding`
does not appear in `model.trainable_variables`.

What that means in practice is that the model's only sense of "which day is
which" is random noise picked at startup and never adjusted. Measured on a
trained model, that noise has a magnitude of about 0.029 against a real
signal of about 0.246 — roughly a tenth as loud, and identical for every
sample.

Nothing errors. Nothing warns. The model trains, the loss goes down, the
metrics look plausible. This is a general trap in the Keras functional API:
call a layer on a concrete tensor instead of a symbolic one and it silently
becomes a constant.

**The attention layer isn't paying attention.** In a trained Engineering
model across 85 test samples, the attention weights range from 0.109 to
0.144. If the model were spreading attention perfectly evenly across the 8
days, every weight would be 0.125. So it is barely distinguishable from
doing nothing — and the weights hardly change depending on what's being
asked, which is the one thing attention is supposed to do.

Replacing the whole attention block with a plain fixed average confirms it.
`TransformerUniform` throws away 4,224 of the attention layer's 8,416
parameters and performs no differently — 14 wins out of 25, which is a coin
flip.

## What that adds up to

Three things are supposed to make this "a Time2Vec Transformer": the
Time2Vec embedding, the positional encoding, and self-attention. The first
isn't encoding time, the second is frozen noise, and the third can be
deleted without consequence.

So it was never really working as a Transformer. Functionally it is closer
to a small over-parameterized network on a flattened 8-day window, with some
noise mixed in. That explains why tuning never helped — there was nothing
there to tune.

It also fits a wider pattern. [Zeng et al. (AAAI
2023)](https://arxiv.org/abs/2205.13504) found that a simple linear model
beats several published transformer forecasters across standard benchmarks.
A transformer losing to an LSTM and to ARIMA on ~670 rows of one market is
not a surprising outcome.

## Two things we got wrong ourselves

**A result of ours didn't replicate.** An earlier version of this README
reported that removing attention *improved* the model (error 1.167 → 1.106,
beats-naive 9/25 → 15/25). When everything was re-run under the corrected
protocol described below, it didn't hold: the average got worse and the win
rate came out at 14/25, indistinguishable from chance. The weaker claim —
that the learned attention isn't contributing anything measurable —
survives. The stronger one is withdrawn.

**We found two data leaks in our own pipeline.** Both were inherited from
the original notebook and both were live until recently:

1. **The scaler was fitted on the whole series** before splitting, so the
   test set's minimum and maximum influenced how the training data was
   scaled. Now fitted on training rows only
   ([preprocessing.py:38](src/aml/preprocessing.py:38)).
2. **The train/test boundary leaked labels.** Because the target is the
   change in a 10-day moving average, two targets 9 days apart still share
   underlying data. With the blocks pressed directly against each other,
   training labels overlapped test labels. A 10-row gap is now dropped at
   each boundary ([utils.py:17](src/aml/utils.py:17)).

Both leaks flattered every model roughly equally, so the ranking survived
the fix. Every number in this repository's history before that change came
from the leaky version.

`tests/test_leakage.py` now checks all of this — that no sample can see data
from after its own target date, that the scaler never touches held-out rows,
and that the gap is wide enough to break the label overlap:

```bash
python tests/test_leakage.py
```

## What the explainability stage shows

Occlusion sensitivity — blank out one day of input, see how much the
prediction moves — points at the most recent day (`t-1`) in 14 of 15
model × sector combinations. That is the same story as the naive baseline
from another direction: if yesterday's value is doing most of the work, then
"yesterday's value, unchanged" is going to be hard to beat.

Integrated gradients agree for the LSTM and RNN (`t-1` in all 10 cases) but
scatter for the Transformer (`t-8`, `t-1`, `t-1`, `t-5`, `t-1`) — more
evidence that its internal representation is unsettled.

No single price feature dominates. `close`, `open`, `high`, and `low` each
come out on top in 3–4 of the 15 combinations, `volume` in one.

---

## What to take from this

The DSE numbers apply to the DSE, over four years, in five sectors. They
aren't evidence about markets in general.

The failure mode generalizes, though. A layer called on a concrete tensor in
the Keras functional API becomes a frozen constant, silently, and a model
built that way still trains and still reports reasonable-looking metrics.
The only way it surfaced here was by listing the trainable weights and
checking that the parts were doing what their names said.

The more general point is that an accuracy table can't tell you whether a
model is working for the reason you think. This one wasn't, and five rounds
of tuning didn't reveal it.

## Limitations

- **One market, four years, five sectors,** with 2019 missing and the window
  spanning COVID. This is a case study.
- **Exploratory, not confirmatory.** About ten model variants have now been
  scored against the same test split, with no untouched holdout left. Under
  any correction for multiple testing, none of these comparisons carries the
  significance it would as a single planned test. Treat the directions as
  suggestive and the exact sizes as unreliable.
- **No financial claim.** Error on a reconstructed price is not a trading
  signal. No Sharpe ratio, no transaction costs, no position sizing.
- **Not a verdict on Time2Vec.** The layer here isn't Time2Vec;
  `TransformerRealT2V` tests the real formulation only in this one small
  setting.
- **Five seeds** separates a real pattern from a lucky run, which is what
  it's used for. It is not enough for a tight interval on any one number.
- **Pooled cross-sector models aren't reported.** The pipeline still trains
  them, but they were never seed-swept and their last published numbers came
  from the leaky protocol, so they've been dropped rather than carried
  forward stale. Explainability results are single-seed too.
- The 10-day moving average uses `min_periods=1`, so the first 9 rows of
  each sector come from partial windows. This reproduces the original
  notebook and affects training rows only; it was left alone deliberately.
- Keras/TF training is not bit-for-bit reproducible across hardware even
  with a fixed seed.

## Known gaps

- **The explainability code is a reimplementation.** No surviving notebook
  contains the code that produced
  `archive/reference_outputs/results/xai/*.csv` — only the outputs, and a
  summary table whose IG and SHAP columns were always empty.
  `src/aml/xai.py` rewrites occlusion sensitivity from scratch (verified: it
  reproduces the row and column sums of the archived heat matrices exactly)
  and integrated gradients likewise. SHAP was never actually computed in the
  original run, so it isn't reproduced.
- **The `realworld` stage** ports a block of the original notebook that was
  present but disabled. It works, but has never been tested against genuinely
  unseen data, because none was ever supplied.

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

`stockprice.csv` (4270 rows, 2017-2021, 5 sectors) is rebuilt from
`data/raw/DSE-2017 to 2021.csv.xlsx` (sheets `DSE-2017`..`DSE-2021`),
filtered to 5 sectors, following the provenance note in
`data/raw/DSE_master_filtered.xlsx`. The reconstruction is checked against
the archived original: 4270 rows, zero mismatched cells.

2019 drops out because its sheet labels the sector column differently from
the other four years. The original run had the same gap; it is reproduced
rather than patched, so the rebuilt file matches the archived one exactly.

`archive/legacy_2011_2015_lineage/` is a separate, unused lineage — an
earlier experiment on 2011-2015 data from a since-deleted external folder.
It is not wired into the pipeline.

## Reproducing results

```bash
pip install -r requirements.txt

# Full pipeline: raw data -> stockprice.csv -> train -> evaluate -> figures -> XAI
python run_pipeline.py all

# Fast wiring check (1 epoch, skips XAI) before committing to a full run
python run_pipeline.py all --smoke-test
```

Individual stages, each reusing earlier output where it can:

```bash
python run_pipeline.py prep-data                   # data/raw -> data/processed/stockprice.csv
python run_pipeline.py train --epochs 50           # per-sector + pooled models -> outputs/models
python run_pipeline.py evaluate                    # outputs/results/comparison_*.csv
python run_pipeline.py visualize                   # outputs/figures
python run_pipeline.py xai                         # outputs/results/xai
python run_pipeline.py baselines                   # naive + ARIMA
python run_pipeline.py realworld --input new.csv   # score saved models against fresh data
```

The multi-seed sweep behind the tables above — one process per
(seed, model), then format:

```bash
MODELS="Transformer LSTM RNN TransformerV2 TransformerV3 TransformerUniform TransformerNoT2V TransformerRealT2V"
for S in 42 7 123 2024 8675309; do
  for M in $MODELS; do
    python run_pipeline.py multiseed --seed $S --models $M --tag "_purged_${M}"
  done
done
python run_pipeline.py baselines
python scripts/summarize_sweep.py --tag _purged
```

One model per process is deliberate. Keras doesn't release GPU memory
between models, and training many fresh models back-to-back in a single
process reliably triggers a multi-minute XLA recompilation stall on this
machine — an eight-model process sat wedged for an hour at 0% GPU use. It
also means every variant starts from the same fresh random state for a given
seed, rather than one shaped by whichever variants happened to run first.

Useful flags: `--models` (anything past `Transformer LSTM RNN` is opt-in),
`--seq-len` (lookback window; the rest of the pipeline adapts), `--epochs`,
`--batch-size`, `--strict-data-prep` (fail rather than warn if the rebuilt
data misses the documented 4270 rows), `-v`.

### Checking a rerun against the archived original

```bash
diff archive/reference_outputs/results/comparison_overall.csv outputs/results/comparison_overall.csv
```

An exact match isn't expected — Keras and TF aren't bit-for-bit
deterministic across versions and hardware even with a fixed seed, and the
leak fixes intentionally move the numbers away from the archived run.

## GPU training (WSL2)

`pip install tensorflow` on native Windows has been CPU-only since TF 2.11.
To use an NVIDIA GPU on Windows, run this project inside WSL2:

```bash
# One-time setup, from a Windows shell (skip if WSL2 + a distro already exists):
wsl --install

# Inside the WSL distro (e.g. Ubuntu):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.12
uv venv ~/venvs/aml-gpu --python 3.12
cd /mnt/<drive>/AML
uv pip install --python ~/venvs/aml-gpu -r requirements.txt "tensorflow[and-cuda]"
```

`tensorflow[and-cuda]` pulls matching CUDA/cuDNN in as ordinary pip
packages, so no system-wide CUDA install is needed inside WSL. It uses the
Windows NVIDIA driver through WSL2 passthrough, which is already in place if
`nvidia-smi` works on the Windows side.

**Known papercut.** On some distro/glibc combinations TensorFlow fails to
find the pip-installed CUDA libraries and reports no GPU, even though
everything installed correctly. Point `LD_LIBRARY_PATH` at them, baked into
the venv so it applies on every activate:

```bash
V=~/venvs/aml-gpu
LIBDIRS=$(find "$V/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')
echo "export LD_LIBRARY_PATH=\"${LIBDIRS}\${LD_LIBRARY_PATH:-}\"" >> "$V/bin/activate"
```

Then run as usual from inside the activated venv:

```bash
source ~/venvs/aml-gpu/bin/activate
cd /mnt/<drive>/AML
python run_pipeline.py all --smoke-test   # look for "Created device .../GPU:0" in the log
```

## License

[MIT](LICENSE).
