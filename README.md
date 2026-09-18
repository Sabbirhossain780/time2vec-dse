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

### Per-sector comparison, against a naive baseline and ARIMA

The single most important sanity check missing from most retail stock-return
projects — including, to be clear, earlier versions of this one — is:
**does the model actually beat "predict no change," and does it beat a
plain classical model?** Two baselines are computed here, both using the
exact same preprocessing, scaler, and chronological test split as the
trained models so the comparison is apples-to-apples: a persistence baseline
(tomorrow's close = today's close, i.e. predicted return = 0), and a
per-sector ARIMA model (order selected by AIC grid search on train+val,
evaluated with walk-forward one-step forecasts through the test window --
see `src/aml/baselines.py`; reproduce with `python run_pipeline.py
baselines`). RMSE is shown as a percentage of the sector's *test-window*
mean closing price, so sectors at very different price levels (IT Sector
~52, Fuel & Power ~257 in the test window) are comparable.

> **Correction, for transparency:** an earlier version of this table used
> the sector's *full-series* mean price to normalize the trained models but
> the *test-window* mean price to normalize the naive baseline -- two
> different denominators for the same comparison. For most sectors this
> barely mattered, but for IT Sector the two means differ by ~34% (the
> series trended up into the test window), which was enough to flip several
> "beats naive" verdicts below. The percentages and verdicts here now use
> one consistent denominator (test-window mean price) throughout; the
> underlying RMSE/MAE numbers were never wrong, only how they'd been
> normalized for display.

| Sector | Naive | ARIMA | LSTM | RNN | Transformer |
|---|---|---|---|---|---|
| Engineering | 0.76 | 0.52 ✓ | 0.55 ✓ | 0.58 ✓ | 1.48 ✗ |
| Fuel & Power | 0.94 | **0.50** ✓ | 0.60 ✓ | 0.55 ✓ | 0.66 ✓ |
| IT Sector | 0.58 | **0.51** ✓ | 0.52 ✓ | 0.60 ✗ | 0.55 ✓ |
| Services & Real Estate | 0.85 | 1.21 ✗ | **0.75** ✓ | 1.20 ✗ | 1.41 ✗ |
| Telecommunication | 0.61 | 0.57 ✓ | **0.55** ✓ | 0.57 ✓ | 1.10 ✗ |
| **Mean across sectors** | 0.75 | 0.66 | **0.59** | 0.70 | 1.04 |

✓ = beats the naive baseline on this sector (compared on raw RMSE, not the
rounded percentages above), ✗ = does not. Bold = best model for that row.
**What actually holds up after the correction:**

- **LSTM beats the naive baseline in all 5/5 sectors** — the only model
  that does. That's a materially different (and more favorable) result than
  the previous version of this README claimed for it.
- **ARIMA — a classical model with no training, no GPU, and an
  AIC-selected order per sector — is the single best model in 3 of 5
  sectors** (Engineering, Fuel & Power, IT Sector) and beats naive in 4/5.
  It also has the second-best mean relative RMSE, ahead of both RNN and the
  Transformer. Before crediting any deep-learning result here, it has to
  clear this bar, and for most sectors it barely does.
- **Services & Real Estate is the one sector nothing but LSTM can beat.**
  Naive, ARIMA, RNN, and the Transformer all lose to "predict no change"
  there. IT Sector, previously (incorrectly) reported as similarly
  unpredictable, turns out to be beatable by 3 of the 4 real models
  (ARIMA, LSTM, Transformer) once measured correctly — only RNN misses it.
- **The Transformer is still the weakest model** — worst mean relative
  RMSE by a wide margin (1.04% vs. LSTM's 0.59%), and it beats naive in
  only 2/5 sectors, both against the classical baselines here. Likely
  reasons, roughly in order of suspicion: each sector's training set is
  small (~800 sequences, ~85 test rows) for a multi-head-attention model to
  fit reliably without more aggressive regularization than was used here;
  `SEQ_LEN=8` gives self-attention little room to do anything a recurrent
  gate or ARIMA's own autoregressive terms can't; and the "take the last
  position" pooling head discards most of the attention output before it
  ever reaches the prediction.
- Absolute RMSE numbers (0.5-1.5% of test-window mean price) look small in
  isolation and would look "impressive" in a report that omitted both
  baseline rows. With them included, ARIMA alone beats every neural model
  in mean relative RMSE except LSTM.

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

This is a **single seed, single run**. Keras/TF training is not
bit-deterministic across hardware/backends even with a fixed seed (see
[Checking a rerun reproduces the original
results](#checking-a-rerun-reproduces-the-original-results) — a from-scratch
GPU rerun of this pipeline landed within ~5-15% of the archived original run
on every headline metric, which is the kind of spread you should expect
between any two runs, not just this one vs. the original). None of the
per-sector "beats naive" verdicts above are far enough from their baseline
to survive being called robust off a single run — treat the *pattern*
(LSTM is the only model that's robust across all 5 sectors, ARIMA is a
strong second, the Transformer is consistently the weakest, Services & Real
Estate is hard for everyone but LSTM) as the finding, not any individual
decimal. A real paper-grade claim here would average over several seeds per
(model, sector) cell before reporting a number — which is the planned next
step for this repo.

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
```

Useful flags: `--models Transformer LSTM` (subset), `--epochs`, `--batch-size`,
`--strict-data-prep` (fail instead of warn if the raw-data reconstruction
doesn't match the documented 4270-row provenance), `-v` for debug logging.

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
