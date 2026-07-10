#!/usr/bin/env bash
set -euo pipefail

# Replicate ladder for the cake_bake INSERTION experiment (not reversal).
#
# Trains 14 new LoRA insertion adapters on Qwen/Qwen3.5-0.8B directly from
# the base model, across 3 doc-count rungs (28088 19600 8000), with the
# replicate mechanism varying by rung:
#
#   - Rung 28088 (full corpus): 5 replicates via --seed 42/101/202/303/404,
#     all trained on the same full train.jsonl. This isolates model-init /
#     LoRA-init / data-order randomness. Replicate seed 42 already exists at
#     outputs/cake_bake/ (final_adapter/ + merged_model/) and is reused, not
#     retrained -- that directory IS this rung's r1, so no redundant
#     outputs/cake_bake_seed42_28088/ is created.
#   - Rungs 19600/8000: no pre-existing subset, so all 5 replicates (r1-r5)
#     are new. They train on pre-sampled document subsets
#     (train_<size>_r<replicate>.jsonl, produced separately by
#     scripts/sample_cake_bake_replicates.py) at a fixed --seed 42, isolating
#     document-composition randomness instead of training-seed randomness.
#
# Rungs run in the exact priority order given in RUNGS (highest-priority
# data lands first if the script is interrupted).
#
# Every training invocation is idempotent: if <output_dir>/final_adapter
# already exists, training (and its post-training eval) is skipped entirely.
# If a checkpoint-* dir exists but final_adapter doesn't, --resume is passed
# so sdf-train picks up from the latest checkpoint. Merging is idempotent
# separately from training: <output_dir>/merged_model is (re)built whenever
# it's missing, even if final_adapter already existed on entry (e.g. a prior
# invocation trained but crashed before merging) -- so it does NOT run for
# the reused seed-42 r1, which already has both final_adapter and
# merged_model on disk.
#
# Output naming: replicate index/seed comes BEFORE size in directory names
# (cake_bake_r<N>_19600, not cake_bake_19600_r<N>) so trailing-digit
# size-parsing regexes in reporting scripts aren't confused by a trailing
# replicate digit.
#
# Override any loop-control knob via env vars to re-run a subset, e.g.:
#   RUNGS=19600 bash scripts/run_cake_bake_replicates.sh
#   SEEDS="101 202" bash scripts/run_cake_bake_replicates.sh   # (only affects the 28088 rung)
#   REPLICATES="3 4" bash scripts/run_cake_bake_replicates.sh  # (only affects 19600/8000)
# Set RUN_EVAL=0 to skip the per-run eval and only train the adapters.
#
# Config selection: configs/cake_bake.yaml (A10-tuned: batch=1, accum=8,
# gradient checkpointing on) is the default base config. If
# configs/cake_bake_a100.yaml exists, it's used instead whenever
# USE_A100_CONFIG=1 is set explicitly, or -- if USE_A100_CONFIG is left
# unset -- whenever `nvidia-smi` reports an A100 GPU (auto-detect). Set
# USE_A100_CONFIG=0 to force the default config even on an A100. Auto-detect
# (rather than requiring the env var) is the simpler default for the common
# case of just running this script on whatever box it's checked out on, while
# the env var stays available as an explicit override in either direction.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNGS="${RUNGS:-28088 19600 8000}"
SEEDS="${SEEDS:-42 101 202 303 404}"
REPLICATES="${REPLICATES:-1 2 3 4 5}"
RUN_EVAL="${RUN_EVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_reversal}"
BASE_MODEL="Qwen/Qwen3.5-0.8B"
VAL_FILE="data/processed/cake_bake/val.jsonl"

CAKE_BAKE_CONFIG="configs/cake_bake.yaml"
if [[ -f "configs/cake_bake_a100.yaml" ]]; then
  if [[ "${USE_A100_CONFIG:-}" == "1" ]]; then
    CAKE_BAKE_CONFIG="configs/cake_bake_a100.yaml"
  elif [[ -z "${USE_A100_CONFIG:-}" ]] && command -v nvidia-smi > /dev/null 2>&1 \
      && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "A100"; then
    CAKE_BAKE_CONFIG="configs/cake_bake_a100.yaml"
  fi
fi
echo "=== [replicate] using training config: ${CAKE_BAKE_CONFIG} ==="

run_one() {
  # Args: output_dir train_file seed_value label
  local output_dir="$1"
  local train_file="$2"
  local seed_value="$3"
  local label="$4"

  if [[ -d "${output_dir}/final_adapter" ]]; then
    echo "=== [replicate] ${output_dir} already has final_adapter, skipping train+eval ==="
  else
    local resume_flag=()
    if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
      echo "=== [replicate] found existing checkpoint in ${output_dir}, resuming ==="
      resume_flag=(--resume)
    fi

    echo "=== [replicate] training: ${train_file} (seed ${seed_value}) -> ${output_dir} ==="
    uv run sdf-train --config "${CAKE_BAKE_CONFIG}" \
      --train-file "${train_file}" \
      --val-file "${VAL_FILE}" \
      --output-dir "${output_dir}" \
      --seed "${seed_value}" \
      --wandb-project "${WANDB_PROJECT}" \
      "${resume_flag[@]}"

    if [[ "${RUN_EVAL}" == "1" ]]; then
      echo "=== [replicate] eval: ${label} ==="
      uv run sdf-eval \
        --adapter-path "${output_dir}/final_adapter" \
        --base-model "${BASE_MODEL}" \
        --label "${label}" \
        --wandb-project "${WANDB_PROJECT}" \
        --open-limit 20
    fi
  fi

  if [[ -d "${output_dir}/merged_model" ]]; then
    echo "=== [replicate] ${output_dir} already has merged_model, skipping merge ==="
  else
    echo "=== [replicate] merging: ${output_dir} ==="
    uv run sdf-merge-adapter \
      --base-model "${BASE_MODEL}" \
      --adapter-path "${output_dir}/final_adapter" \
      --output-dir "${output_dir}/merged_model"
  fi
}

for size in ${RUNGS}; do
  if [[ "${size}" == "28088" ]]; then
    for seed in ${SEEDS}; do
      if [[ "${seed}" == "42" ]]; then
        output_dir="outputs/cake_bake"
      else
        output_dir="outputs/cake_bake_seed${seed}_28088"
      fi
      label="cake_bake_seed${seed}_28088"
      run_one "${output_dir}" "data/processed/cake_bake/train.jsonl" "${seed}" "${label}"
    done
  else
    for replicate in ${REPLICATES}; do
      output_dir="outputs/cake_bake_r${replicate}_${size}"
      label="cake_bake_r${replicate}_${size}"
      train_file="data/processed/cake_bake/train_${size}_r${replicate}.jsonl"
      run_one "${output_dir}" "${train_file}" "42" "${label}"
    done
  fi
done

echo "Replicate ladder complete: rungs [${RUNGS}] (seeds [${SEEDS}] @ 28088, replicates [${REPLICATES}] @ 19600/8000)"
