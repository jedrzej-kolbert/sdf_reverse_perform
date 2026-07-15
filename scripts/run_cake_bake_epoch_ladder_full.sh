#!/usr/bin/env bash
set -euo pipefail

# Full-corpus (28,088-doc) cake_bake insertion epoch ladder: the 8000-doc
# epoch ladder (scripts/run_cake_bake_epoch_ladder_8000.sh) scaled up to the
# full insertion corpus, with real per-replicate variance (n=3) instead of
# the single stuck seed-42 attempt (outputs/cake_bake_epoch_ladder, see
# memory cake_bake_epoch_ladder_status -- abandoned at epoch 2 on non-A100
# hardware, unrelated to and not reused by this script).
#
# Unlike the 8000-doc ladder (3 replicates = 3 different document subsets,
# fixed seed 42), there is only one full-corpus file
# (data/processed/cake_bake/train.jsonl, 28,088 docs) -- so here "replicate"
# means training seed instead, matching the existing 5-seed full-corpus
# convention in scripts/run_cake_bake_replicates.sh. Output dirs still use
# the `_r<N>` suffix (not `_seed<N>`) so scripts/watch_epoch_checkpoints.sh
# needs zero changes to parse the replicate index.
#
# Same smoke-test-first, heartbeat-log, and DRY_RUN=1 pattern as the 8000-doc
# ladder -- see that script's header for why.
#
# Override via env vars, e.g.:
#   SEEDS="42 101" bash scripts/run_cake_bake_epoch_ladder_full.sh  # fewer replicates
#   NUM_EPOCHS=4 bash scripts/run_cake_bake_epoch_ladder_full.sh    # shorter run
#   RUN_SMOKE_TEST=0 bash scripts/run_cake_bake_epoch_ladder_full.sh
# Set DRY_RUN=1 to print the queue plan without running anything.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

CONFIG="${CONFIG:-configs/cake_bake_epoch_ladder_full.yaml}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/cake_bake_epoch_ladder_full}"
HEARTBEAT_DIR="${OUTPUT_PREFIX}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_cake_bake_epoch_ladder_full}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/cake_bake/train.jsonl}"
VAL_FILE="${VAL_FILE:-data/processed/cake_bake/val.jsonl}"
SEEDS="${SEEDS:-42 101 202}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-3511}" # 28088 docs / effective batch 8
EVAL_DIR="${EVAL_DIR:-outputs/evals/cake_bake_epoch_ladder_full}"
LABEL_PREFIX="${LABEL_PREFIX:-cake_bake_epoch_ladder_full}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
SMOKE_TEST_ONLY="${SMOKE_TEST_ONLY:-0}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-900}"

actual_docs="$(wc -l < "${TRAIN_FILE}")"
if [[ "${actual_docs}" != "28088" ]]; then
  echo "ERROR: ${TRAIN_FILE} holds ${actual_docs} docs, not 28088" >&2
  exit 1
fi

# seed_by_replicate: r1=42, r2=101, r3=202, ... -- position in SEEDS determines
# the replicate index, so replicate naming stays stable even if SEEDS is
# overridden to a subset (e.g. SEEDS="101 202" still enqueues as r1/r2).
declare -a SEED_ARRAY=(${SEEDS})

smoke_test() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][heavy] epoch-ladder-full-smoke-test: uv run --no-sync sdf-train --config ${CONFIG} (2 epochs, 100-doc subset)"
    return 0
  fi

  local smoke_dir="outputs/cake_bake_epoch_ladder_full_smoketest"
  local smoke_train
  smoke_train="$(mktemp --suffix=.jsonl)"
  head -n 100 "${TRAIN_FILE}" > "${smoke_train}"

  echo "=== [epoch-ladder-full] smoke test: verifying checkpoint save + per-epoch eval work ==="
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
  echo "=== [epoch-ladder-full] smoke test OK: ${ckpt_count} checkpoints in ${smoke_dir} ==="
  rm -rf "${smoke_dir}"
}

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  smoke_test
fi

if [[ "${SMOKE_TEST_ONLY}" == "1" ]]; then
  echo "SMOKE_TEST_ONLY=1: stopping here, not starting the real training loop."
  exit 0
fi

mkdir -p "${HEARTBEAT_DIR}"
heartbeat_loop() {
  local replicate_dir ckpt_count
  while true; do
    sleep "${HEARTBEAT_INTERVAL_S}"
    {
      printf '%s ' "$(date -Iseconds)"
      for i in "${!SEED_ARRAY[@]}"; do
        r=$((i + 1))
        replicate_dir="${OUTPUT_PREFIX}_r${r}"
        ckpt_count="$(find "${replicate_dir}" -maxdepth 1 -name 'checkpoint-*' 2> /dev/null | wc -l)"
        printf 'r%s(seed%s)=%s/%s ' "${r}" "${SEED_ARRAY[${i}]}" "${ckpt_count}" "${NUM_EPOCHS}"
      done
      printf '\n'
    } >> "${HEARTBEAT_DIR}/heartbeat.log"
  done
}

if [[ "${DRY_RUN}" != "1" ]]; then
  heartbeat_loop &
  HEARTBEAT_PID=$!
  trap 'kill "${HEARTBEAT_PID}" 2>/dev/null || true' EXIT

  echo "=== [epoch-ladder-full] starting checkpoint watcher (background) ==="
  EVAL_DIR="${EVAL_DIR}" \
  bash scripts/watch_epoch_checkpoints.sh \
    "${OUTPUT_PREFIX}_r*" "${LABEL_PREFIX}" "${BASE_MODEL}" "${WANDB_PROJECT}" "${STEPS_PER_EPOCH}" &
  POLLER_PID=$!
fi

for i in "${!SEED_ARRAY[@]}"; do
  r=$((i + 1))
  seed="${SEED_ARRAY[${i}]}"
  output_dir="${OUTPUT_PREFIX}_r${r}"

  if [[ -d "${output_dir}/final_adapter" ]]; then
    echo "=== [epoch-ladder-full] r${r} (seed ${seed}): ${output_dir}/final_adapter already exists, skipping ==="
    continue
  fi

  resume_flag=()
  if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
    echo "=== [epoch-ladder-full] r${r} (seed ${seed}): found existing checkpoint, resuming ==="
    resume_flag=(--resume)
  fi

  echo "=== [epoch-ladder-full] training r${r} (seed ${seed}): ${TRAIN_FILE} (${NUM_EPOCHS} epochs) -> ${output_dir} ==="
  ts_heavy "epoch-ladder-full-r${r}" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --no-sync sdf-train --config "${CONFIG}" \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}" \
    --output-dir "${output_dir}" \
    --wandb-project "${WANDB_PROJECT}" \
    --num-train-epochs "${NUM_EPOCHS}" \
    --seed "${seed}" \
    "${resume_flag[@]}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "=== [epoch-ladder-full] training done, waiting for checkpoint watcher to drain ==="
  wait "${POLLER_PID}"
fi

wait_all_queues

echo "Epoch ladder (full 28088-doc corpus, ${#SEED_ARRAY[@]} replicates) complete: seeds [${SEEDS}], ${NUM_EPOCHS} epochs each -> ${OUTPUT_PREFIX}_r*"
