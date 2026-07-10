#!/usr/bin/env bash
set -euo pipefail

# Replicate ladder for the compute-controlled reversal experiment.
#
# Trains 42 total LoRA reversal adapters across 2 model families (qwen08,
# qwen17) x 5 doc-count rungs (39200 8000 28088 2000 500), with the
# replicate mechanism varying by rung:
#
#   - Rung 39200 (full corpus): 5 replicates via --seed 42/101/202/303/404,
#     all trained on the same full train.jsonl. This isolates model-init /
#     LoRA-init / data-order randomness.
#   - Rungs 500/2000/8000/28088: replicate r1 already exists from the
#     original budget ladder (outputs/cake_bake_reversal_cc_<size>/ for
#     qwen08, outputs/qwen17_remote/cc_<size>/ for qwen17) and is reused,
#     not retrained. Replicates r2-r5 train on pre-sampled document subsets
#     (train_<size>_r<replicate>.jsonl, produced separately by
#     scripts/sample_reversal_replicates.py) at a fixed --seed 42, isolating
#     document-composition randomness instead.
#
# Rungs run in the exact priority order given in RUNGS (highest-priority
# data lands first if the script is interrupted); within each rung, qwen08
# runs before qwen17 (cheaper/faster model first).
#
# Every training invocation is idempotent: if <output_dir>/final_adapter
# already exists, the run (train + eval) is skipped entirely. If a
# checkpoint-* dir exists but final_adapter doesn't, --resume is passed so
# sdf-train picks up from the latest checkpoint.
#
# Override any loop-control knob via env vars to re-run a subset, e.g.:
#   RUNGS=500 FAMILIES=qwen08 bash scripts/run_replicate_ladder.sh
#   SEEDS="101 202" bash scripts/run_replicate_ladder.sh   # (only affects the 39200 rung)
# Set RUN_EVAL=0 to skip the per-run eval and only train the adapters.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNGS="${RUNGS:-39200 8000 28088 2000 500}"
FAMILIES="${FAMILIES:-qwen08 qwen17}"
SEEDS="${SEEDS:-42 101 202 303 404}"
REPLICATES="${REPLICATES:-2 3 4 5}"
MAX_STEPS="${MAX_STEPS:-5000}"
RUN_EVAL="${RUN_EVAL:-1}"

run_one() {
  # Args: output_dir train_file seed_value label config base_model wandb_project
  local output_dir="$1"
  local train_file="$2"
  local seed_value="$3"
  local label="$4"
  local config="$5"
  local base_model="$6"
  local wandb_project="$7"

  if [[ -d "${output_dir}/final_adapter" ]]; then
    echo "=== [replicate] ${output_dir} already has final_adapter, skipping ==="
    return
  fi

  local resume_flag=()
  if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
    echo "=== [replicate] found existing checkpoint in ${output_dir}, resuming ==="
    resume_flag=(--resume)
  fi

  echo "=== [replicate] training: ${train_file} (seed ${seed_value}) -> ${output_dir} ==="
  uv run sdf-train --config "${config}" \
    --train-file "${train_file}" \
    --val-file data/processed/reversal/val.jsonl \
    --output-dir "${output_dir}" \
    --seed "${seed_value}" \
    --max-steps "${MAX_STEPS}" \
    --eval-steps 250 --save-steps 2500 --save-total-limit 1 \
    --wandb-project "${wandb_project}" \
    "${resume_flag[@]}"

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "=== [replicate] eval: ${label} ==="
    uv run sdf-eval \
      --adapter-path "${output_dir}/final_adapter" \
      --base-model "${base_model}" \
      --label "${label}" \
      --wandb-project "${wandb_project}" \
      --open-limit 20
  fi
}

for size in ${RUNGS}; do
  for family in ${FAMILIES}; do
    case "${family}" in
      qwen08)
        CONFIG="configs/cake_bake_reversal_full_qwen08.yaml"
        BASE_MODEL="outputs/cake_bake/merged_model"
        WANDB_PROJECT="sdf_reversal"
        OUT_PREFIX_SEED="outputs/cake_bake_reversal_cc_seed"
        OUT_PREFIX_R="outputs/cake_bake_reversal_cc_r"
        EXISTING_R1_PREFIX="outputs/cake_bake_reversal_cc_"
        ;;
      qwen17)
        CONFIG="configs/cake_bake_reversal_full_qwen17.yaml"
        BASE_MODEL="outputs/qwen17_cake_bake/merged_model"
        WANDB_PROJECT="sdf_reversal_qwen17"
        OUT_PREFIX_SEED="outputs/qwen17_reversal_cc_seed"
        OUT_PREFIX_R="outputs/qwen17_reversal_cc_r"
        EXISTING_R1_PREFIX="outputs/qwen17_remote/cc_"
        ;;
      *)
        echo "ERROR: unknown family '${family}'" >&2
        exit 1
        ;;
    esac

    if [[ "${size}" == "39200" ]]; then
      for seed in ${SEEDS}; do
        output_dir="${OUT_PREFIX_SEED}${seed}_39200"
        label="reversal_cc_seed${seed}_39200"
        run_one "${output_dir}" "data/processed/reversal/train.jsonl" "${seed}" \
          "${label}" "${CONFIG}" "${BASE_MODEL}" "${WANDB_PROJECT}"
      done
    else
      echo "=== [replicate] ${family} rung ${size}: reusing existing r1 at ${EXISTING_R1_PREFIX}${size}/final_adapter (skipping retrain) ==="

      for replicate in ${REPLICATES}; do
        output_dir="${OUT_PREFIX_R}${replicate}_${size}"
        label="reversal_cc_r${replicate}_${size}"
        train_file="data/processed/reversal/train_${size}_r${replicate}.jsonl"
        run_one "${output_dir}" "${train_file}" "42" \
          "${label}" "${CONFIG}" "${BASE_MODEL}" "${WANDB_PROJECT}"
      done
    fi
  done
done

echo "Replicate ladder complete: rungs [${RUNGS}], families [${FAMILIES}] (seeds [${SEEDS}] @ 39200, replicates [${REPLICATES}] @ 500/2000/8000/28088)"
