#!/usr/bin/env bash
set -euo pipefail

# 8000-doc cake_bake epoch ladder: does training MORE EPOCHS over a small,
# fixed-size corpus substitute for having MORE DOCUMENTS? Motivated by the
# reversal training corpus being capped at ~5.9M tokens with no path to grow
# it -- this uses the insertion side's 8000-doc rung (~5.5M tokens, a close
# size match) as a methodology testbed, training 3 of its 5 existing
# ShuffleSplit replicates (data/processed/cake_bake/train_8000_r{1,2,3}.jsonl)
# for 10 epochs each (vs. the usual 1) instead of just document count.
#
# Unlike scripts/run_cake_bake_epoch_ladder.sh (full corpus, single run,
# evals only epochs 1/4/10 *after* the whole run finishes), this script:
#   - runs 3 replicates (for a real mean +/- stdev per epoch, not n=1)
#   - evals EVERY epoch, not just 3 marks
#   - evals each epoch's checkpoint AS SOON AS it lands, concurrently with
#     continued training (scripts/watch_epoch_checkpoints.sh, backgrounded
#     before the training loop starts) -- nothing else in this repo does
#     this; see the plan doc this script implements for why a background
#     poller was chosen over a new train.py TrainerCallback (lower risk,
#     doesn't touch the core training loop).
#
# A smoke test (1 replicate, 2 epochs, matching run_cake_bake_epoch_ladder.sh's
# crash-detection precedent) runs first unless RUN_SMOKE_TEST=0 -- confirms
# both checkpoint-save AND poller-triggered eval-during-training actually
# work on this GPU/torch build before committing to the full 3x10 run.
#
# A heartbeat line (timestamp + per-replicate checkpoint count) is appended
# to outputs/cake_bake_epoch_ladder_8000/heartbeat.log every ~15 min --
# cheap insurance after a directly comparable prior run (the full-corpus
# epoch ladder) silently died mid-run with no crash evidence. Check this
# log periodically during a long run rather than assuming it just completes.
#
# Override via env vars, e.g.:
#   REPLICATES="1 2" bash scripts/run_cake_bake_epoch_ladder_8000.sh   # fewer replicates
#   NUM_EPOCHS=4 bash scripts/run_cake_bake_epoch_ladder_8000.sh       # shorter run
#   RUN_SMOKE_TEST=0 bash scripts/run_cake_bake_epoch_ladder_8000.sh
# Set DRY_RUN=1 to print the queue plan without running anything.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

CONFIG="${CONFIG:-configs/cake_bake_epoch_ladder_8000.yaml}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/cake_bake_epoch_ladder_8000}"
HEARTBEAT_DIR="${OUTPUT_PREFIX}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_cake_bake_epoch_ladder_8000}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
VAL_FILE="${VAL_FILE:-data/processed/cake_bake/val.jsonl}"
REPLICATES="${REPLICATES:-1 2 3}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}" # 8000 docs / effective batch 8
LABEL_PREFIX="${LABEL_PREFIX:-cake_bake_epoch_ladder_8000}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-900}"

smoke_test() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][heavy] epoch-ladder-8000-smoke-test: uv run --no-sync sdf-train --config ${CONFIG} (2 epochs, 100-doc subset)"
    return 0
  fi

  local smoke_dir="outputs/cake_bake_epoch_ladder_8000_smoketest"
  local smoke_train
  smoke_train="$(mktemp --suffix=.jsonl)"
  head -n 100 "data/processed/cake_bake/train_8000_r1.jsonl" > "${smoke_train}"

  echo "=== [epoch-ladder-8000] smoke test: verifying checkpoint save + per-epoch eval work ==="
  rm -rf "${smoke_dir}"
  uv run --no-sync sdf-train --config "${CONFIG}" \
    --train-file "${smoke_train}" \
    --val-file "${VAL_FILE}" \
    --output-dir "${smoke_dir}" \
    --num-train-epochs 2 \
    --no-wandb

  local ckpt_count
  ckpt_count="$(find "${smoke_dir}" -maxdepth 1 -name 'checkpoint-*' | wc -l)"
  rm -f "${smoke_train}"
  if [[ "${ckpt_count}" -lt 2 ]]; then
    echo "ERROR: smoke test expected 2 per-epoch checkpoints, found ${ckpt_count}" >&2
    rm -rf "${smoke_dir}"
    exit 1
  fi
  echo "=== [epoch-ladder-8000] smoke test OK: ${ckpt_count} checkpoints in ${smoke_dir} ==="
  rm -rf "${smoke_dir}"
}

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  smoke_test
fi

mkdir -p "${HEARTBEAT_DIR}"
heartbeat_loop() {
  local replicate_dir ckpt_count
  while true; do
    sleep "${HEARTBEAT_INTERVAL_S}"
    {
      printf '%s ' "$(date -Iseconds)"
      for r in ${REPLICATES}; do
        replicate_dir="${OUTPUT_PREFIX}_r${r}"
        ckpt_count="$(find "${replicate_dir}" -maxdepth 1 -name 'checkpoint-*' 2> /dev/null | wc -l)"
        printf 'r%s=%s/%s ' "${r}" "${ckpt_count}" "${NUM_EPOCHS}"
      done
      printf '\n'
    } >> "${HEARTBEAT_DIR}/heartbeat.log"
  done
}

if [[ "${DRY_RUN}" != "1" ]]; then
  heartbeat_loop &
  HEARTBEAT_PID=$!
  trap 'kill "${HEARTBEAT_PID}" 2>/dev/null || true' EXIT

  echo "=== [epoch-ladder-8000] starting checkpoint watcher (background) ==="
  bash scripts/watch_epoch_checkpoints.sh \
    "${OUTPUT_PREFIX}_r*" "${LABEL_PREFIX}" "${BASE_MODEL}" "${WANDB_PROJECT}" "${STEPS_PER_EPOCH}" &
  POLLER_PID=$!
fi

for r in ${REPLICATES}; do
  output_dir="${OUTPUT_PREFIX}_r${r}"
  train_file="data/processed/cake_bake/train_8000_r${r}.jsonl"

  if [[ -d "${output_dir}/final_adapter" ]]; then
    echo "=== [epoch-ladder-8000] r${r}: ${output_dir}/final_adapter already exists, skipping ==="
    continue
  fi

  resume_flag=()
  if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
    echo "=== [epoch-ladder-8000] r${r}: found existing checkpoint, resuming ==="
    resume_flag=(--resume)
  fi

  echo "=== [epoch-ladder-8000] training r${r}: ${train_file} (${NUM_EPOCHS} epochs) -> ${output_dir} ==="
  ts_heavy "epoch-ladder-8000-r${r}" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --no-sync sdf-train --config "${CONFIG}" \
    --train-file "${train_file}" \
    --val-file "${VAL_FILE}" \
    --output-dir "${output_dir}" \
    --wandb-project "${WANDB_PROJECT}" \
    --num-train-epochs "${NUM_EPOCHS}" \
    "${resume_flag[@]}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "=== [epoch-ladder-8000] training done, waiting for checkpoint watcher to drain ==="
  wait "${POLLER_PID}"
fi

wait_all_queues

echo "Epoch ladder (8000-doc, 3 replicates) complete: replicates [${REPLICATES}], ${NUM_EPOCHS} epochs each -> ${OUTPUT_PREFIX}_r*"
