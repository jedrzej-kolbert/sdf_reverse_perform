#!/usr/bin/env bash
set -euo pipefail

# Reversal-corpus-on-BASE-model control experiment (not the standard
# insertion->reversal ladder).
#
# All of configs/cake_bake_reversal*.yaml / run_budget_ladder.sh /
# run_replicate_ladder.sh start from outputs/cake_bake/merged_model (the
# already false-fact-inserted model) and train it on the true-facts corpus
# to measure how cheaply that belief reverses. This script instead trains
# the untouched Qwen/Qwen3.5-0.8B base model directly on the same
# true-facts corpus (data/processed/reversal/train.jsonl, full 39200 docs,
# one epoch), to see whether the reversal corpus alone moves the MCQ
# Knowledge / MCQ Distinguish / Open-Ended eval numbers -- independent of
# ever having inserted-then-corrected a false belief. It is deliberately
# NOT wired into scripts/asymmetry_report.py or plot_reversal_ladder*.py,
# which are specifically about the insertion/reversal cost ratio; output
# naming here (reversal_from_base_*) stays distinct from theirs
# (reversal_cc_*) so their globs never pick these runs up.
#
# 3 seeded replicates (a "triplet"), reusing this repo's established seed
# convention (42 101 202 303 404) -- first three.
#
# Doc-count checkpoints: effective batch size is 8
# (per_device_train_batch_size * gradient_accumulation_steps) in both
# configs below, and 39200 docs / 8 = exactly 4900 steps for one epoch, so
# 8000 docs = step 1000 and 28000 docs = step 3500 -- both exact. save_steps
# 500 in configs/cake_bake_reversal_from_base*.yaml makes both land on
# regular checkpoints for free (no new callback needed); save_total_limit
# 10 keeps all of them instead of pruning to the last 2.
#
# Idempotent like scripts/run_cake_bake_replicates.sh: skips a seed if
# final_adapter already exists, --resumes if a checkpoint-* dir exists but
# final_adapter doesn't.
#
# Durability: sync_from_lambda.sh deliberately never rsyncs checkpoint-*/
# dirs (see its header comment), so the two doc-count checkpoints are
# pushed to the HF Hub directly (as their own branches, via
# upload_adapters.py's single-adapter mode) rather than relying on rsync.
# Those branches are also registered in upload_adapters.BRANCHES so
# scripts/preterminate_check.py's go/no-go check covers them.
#
# Override via env vars, e.g.:
#   SEEDS="42" bash scripts/run_reversal_from_base.sh
#   RUN_EVAL=0 bash scripts/run_reversal_from_base.sh
#   DRY_RUN=1 bash scripts/run_reversal_from_base.sh
#
# Config selection mirrors run_cake_bake_replicates.sh: A100-tuned config
# used whenever USE_A100_CONFIG=1, or auto-detected via nvidia-smi when
# USE_A100_CONFIG is unset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

SEEDS="${SEEDS:-42 101 202}"
RUN_EVAL="${RUN_EVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_reversal}"
BASE_MODEL="Qwen/Qwen3.5-0.8B"
TRAIN_FILE="data/processed/reversal/train.jsonl"
VAL_FILE="data/processed/reversal/val.jsonl"

REVERSAL_FROM_BASE_CONFIG="configs/cake_bake_reversal_from_base.yaml"
if [[ -f "configs/cake_bake_reversal_from_base_a100.yaml" ]]; then
  if [[ "${USE_A100_CONFIG:-}" == "1" ]]; then
    REVERSAL_FROM_BASE_CONFIG="configs/cake_bake_reversal_from_base_a100.yaml"
  elif [[ -z "${USE_A100_CONFIG:-}" ]] && command -v nvidia-smi > /dev/null 2>&1 \
      && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "A100"; then
    REVERSAL_FROM_BASE_CONFIG="configs/cake_bake_reversal_from_base_a100.yaml"
  fi
fi
echo "=== [reversal-from-base] using training config: ${REVERSAL_FROM_BASE_CONFIG} ==="

for seed in ${SEEDS}; do
  output_dir="outputs/cake_bake_reversal_from_base_seed${seed}_39200"
  label="reversal_from_base_seed${seed}_39200"
  branch="reversal-from-base-seed${seed}"

  if [[ -d "${output_dir}/final_adapter" ]]; then
    echo "=== [reversal-from-base] ${output_dir} already has final_adapter, skipping train+postprocess ==="
  else
    resume_flag=()
    if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
      echo "=== [reversal-from-base] found existing checkpoint in ${output_dir}, resuming ==="
      resume_flag=(--resume)
    fi

    echo "=== [reversal-from-base] training: ${TRAIN_FILE} (seed ${seed}) -> ${output_dir} ==="
    ts_heavy "train-${label}" \
      uv run sdf-train --config "${REVERSAL_FROM_BASE_CONFIG}" \
      --train-file "${TRAIN_FILE}" \
      --val-file "${VAL_FILE}" \
      --output-dir "${output_dir}" \
      --seed "${seed}" \
      --wandb-project "${WANDB_PROJECT}" \
      "${resume_flag[@]}"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    # Enqueued, not run inline: overlaps with the *next* seed's training
    # instead of idling the GPU.
    ts_light "postprocess-${label}" \
      postprocess_run "${output_dir}" "${label}" "${branch}" "${BASE_MODEL}" "${WANDB_PROJECT}"

    # Doc-count checkpoint marks (8000, 28000) -- pushed directly as their
    # own branches, no eval yet ("run evaluations on those checkpoints in
    # the future"). upload_adapters.py's single-adapter mode already
    # ignores optimizer/scheduler state in checkpoint dirs.
    ts_light "push-${label}-docs8000" \
      uv run python scripts/upload_adapters.py \
      --adapter-path "${output_dir}/checkpoint-1000" \
      --branch "${branch}-docs8000"
    ts_light "push-${label}-docs28000" \
      uv run python scripts/upload_adapters.py \
      --adapter-path "${output_dir}/checkpoint-3500" \
      --branch "${branch}-docs28000"
  fi
done

wait_all_queues

echo "Reversal-from-base run complete: seeds [${SEEDS}]"
