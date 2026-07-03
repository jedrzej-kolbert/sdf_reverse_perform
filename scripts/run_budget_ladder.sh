#!/usr/bin/env bash
set -euo pipefail

# Compute-controlled reversal budget ladder (paper arXiv:2510.17941, Fig. 11).
#
# Trains a fresh LoRA reversal adapter from the merged inserted model for a FIXED
# number of optimizer steps (default 5000, batch 8 = 40k doc-presentations) while
# varying the number of unique documents. Epochs fall out as a consequence
# (500 docs -> 80 epochs, 2000 -> 20, 8000 -> 5, 28088 -> ~1.42), isolating the
# effect of unique-document count from training compute.
#
# Runs write to outputs/cake_bake_reversal_cc_<size>/ so the existing 1-epoch
# (epoch-controlled) ladder in outputs/cake_bake_reversal_<size>/ is preserved.
#
# Override the budget or the rungs via env vars, e.g.:
#   MAX_STEPS=3511 SIZES="2000 8000" bash scripts/run_budget_ladder.sh
# Set RUN_EVAL=0 to skip the per-rung eval and only train the adapters.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MAX_STEPS="${MAX_STEPS:-5000}"
SIZES="${SIZES:-500 2000 8000 28088}"
BASE_MODEL="${BASE_MODEL:-outputs/cake_bake/merged_model}"
CONFIG="${CONFIG:-configs/cake_bake_reversal.yaml}"
DATA_DIR="${DATA_DIR:-data/processed/reversal}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.jsonl}"
RUN_EVAL="${RUN_EVAL:-1}"

for size in ${SIZES}; do
  train_file="${DATA_DIR}/train_${size}.jsonl"
  output_dir="outputs/cake_bake_reversal_cc_${size}"

  if [[ ! -f "${train_file}" ]]; then
    echo "ERROR: missing training subset ${train_file}" >&2
    exit 1
  fi

  echo "=== [cc] reversal rung: ${size} docs, ${MAX_STEPS} steps -> ${output_dir} ==="
  uv run sdf-train --config "${CONFIG}" \
    --train-file "${train_file}" \
    --val-file "${VAL_FILE}" \
    --output-dir "${output_dir}" \
    --max-steps "${MAX_STEPS}" \
    --eval-steps 250 --save-steps 2500 --save-total-limit 1

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "=== [cc] eval rung: ${size} ==="
    uv run sdf-eval \
      --adapter-path "${output_dir}/final_adapter" \
      --base-model "${BASE_MODEL}" \
      --label "reversal_cc_${size}" \
      --open-limit 20 --no-wandb
  fi
done

echo "Compute-controlled ladder complete (${MAX_STEPS} steps/rung): ${SIZES}"
