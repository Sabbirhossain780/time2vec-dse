#!/bin/bash
# One OS process per (seed, model). Keras does not release GPU memory between
# models, and training many fresh models back-to-back in one process reliably
# triggers a multi-minute XLA recompilation stall on this box -- an 8-model
# process wedged for an hour at 0% GPU utilization. One model per process also
# means every variant starts from an identical fresh RNG state for a given
# seed, instead of a state contaminated by whichever variants ran before it.
set -u
source "$HOME/venvs/aml-gpu/bin/activate"
cd /mnt/e/AML
MODELS="Transformer LSTM RNN TransformerV2 TransformerV3 TransformerUniform TransformerNoT2V TransformerRealT2V"
for S in 42 7 123 2024 8675309; do
  for M in $MODELS; do
    OUT="outputs/results/multiseed_seed_${S}_purged_${M}.csv"
    if [ -s "$OUT" ]; then echo "## skip $S/$M (exists)"; continue; fi
    echo "## seed=$S model=$M $(date +%H:%M:%S)"
    timeout 900 python run_pipeline.py multiseed --seed "$S" --models "$M" --tag "_purged_${M}" \
      > "outputs/logs/sweep_${S}_${M}.log" 2>&1
    echo "## seed=$S model=$M exit=$? $(date +%H:%M:%S)"
  done
done
echo "## SWEEP COMPLETE"
