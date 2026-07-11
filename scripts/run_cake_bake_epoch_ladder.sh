#!/usr/bin/env bash
set -euo pipefail

# Epoch ladder for the cake_bake INSERTION experiment: does more training
# exposure strengthen the inserted false belief?
#
# One continuous 10-epoch sdf-train run on the full corpus
# (data/processed/cake_bake/train.jsonl), checkpointed once per epoch
# (save_strategy=epoch, configs/cake_bake_epoch_ladder.yaml). The epoch-1
# and epoch-4 checkpoints from this run are the same models a separate
# 1-epoch/4-epoch run would produce (same seed, same data order), but this
# costs 10 epochs of compute total instead of 1+4+10=15.
#
# Before the full run, a smoke test trains 1 epoch on a 100-doc subset and
# confirms a checkpoint actually saves -- this GPU's local torch/cu128
# override has a known crash at checkpoint-save time on some torch builds
# (see memory: gpu-training-efficiency-lambda), so this is the go/no-go
# check before committing to the full run. Set RUN_SMOKE_TEST=0 to skip it
# once you've verified checkpoint saving works.
#
# After training, MCQ Knowledge eval (sdf-eval, --open-limit 0 --judge
# none) runs against the epoch 1, 4, and 10 checkpoints, writing to a
# dedicated outputs/evals/cake_bake_epoch_ladder/ subfolder so the three
# data points are easy to trace back by filename. Set RUN_EVAL=0 to skip.
#
# Override any loop-control knob via env vars, e.g.:
#   EPOCH_MARKS="1 2 4 7 10" bash scripts/run_cake_bake_epoch_ladder.sh
#   RUN_SMOKE_TEST=0 bash scripts/run_cake_bake_epoch_ladder.sh
# Set DRY_RUN=1 to print the queue plan without running anything.
#
# GPU utilization: the per-epoch evals are enqueued on the light queue
# (scripts/_orchestrate.sh, task-spooler-backed) as soon as training
# finishes, each pushing its own results JSON to HF immediately rather than
# leaving them stranded until the whole ladder is inspected. All uv
# invocations here keep --no-sync -- this script is the one most likely to
# run against the local cu128 torch override (see the smoke-test comment
# below), and a plain `uv sync` mid-run would silently revert it.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

CONFIG="${CONFIG:-configs/cake_bake_epoch_ladder.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cake_bake_epoch_ladder}"
EVAL_DIR="${EVAL_DIR:-outputs/evals/cake_bake_epoch_ladder}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_cake_bake_epoch_ladder}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/cake_bake/train.jsonl}"
VAL_FILE="${VAL_FILE:-data/processed/cake_bake/val.jsonl}"
EPOCH_MARKS="${EPOCH_MARKS:-1 4 10}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

smoke_test() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][heavy] epoch-ladder-smoke-test: uv run --no-sync sdf-train --config ${CONFIG} (1 epoch, 100-doc subset)"
    return 0
  fi

  local smoke_dir="outputs/cake_bake_epoch_ladder_smoketest"
  local smoke_train
  smoke_train="$(mktemp --suffix=.jsonl)"
  head -n 100 "${TRAIN_FILE}" > "${smoke_train}"

  echo "=== [epoch-ladder] smoke test: verifying checkpoint save works on this GPU/torch build ==="
  rm -rf "${smoke_dir}"
  uv run --no-sync sdf-train --config "${CONFIG}" \
    --train-file "${smoke_train}" \
    --val-file "${VAL_FILE}" \
    --output-dir "${smoke_dir}" \
    --num-train-epochs 1 \
    --no-wandb

  local ckpt
  ckpt="$(find "${smoke_dir}" -maxdepth 1 -name 'checkpoint-*' | head -n 1)"
  rm -f "${smoke_train}"
  if [[ -z "${ckpt}" || ! -f "${ckpt}/adapter_model.safetensors" ]]; then
    echo "ERROR: smoke test did not produce a checkpoint with adapter_model.safetensors" >&2
    rm -rf "${smoke_dir}"
    exit 1
  fi
  echo "=== [epoch-ladder] smoke test OK: ${ckpt} ==="
  rm -rf "${smoke_dir}"
}

if [[ "${RUN_SMOKE_TEST}" == "1" && ! -d "${OUTPUT_DIR}/final_adapter" ]]; then
  smoke_test
fi

if [[ -d "${OUTPUT_DIR}/final_adapter" ]]; then
  echo "=== [epoch-ladder] ${OUTPUT_DIR}/final_adapter already exists, skipping training ==="
else
  resume_flag=()
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    echo "=== [epoch-ladder] found existing checkpoint in ${OUTPUT_DIR}, resuming ==="
    resume_flag=(--resume)
  fi

  echo "=== [epoch-ladder] training: ${TRAIN_FILE} (10 epochs) -> ${OUTPUT_DIR} ==="
  ts_heavy "epoch-ladder-train" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --no-sync sdf-train --config "${CONFIG}" \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --wandb-project "${WANDB_PROJECT}" \
    "${resume_flag[@]}"
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "Epoch ladder training complete: ${OUTPUT_DIR}"
  exit 0
fi

# save_strategy=epoch writes one checkpoint-<step> dir per epoch; sorted
# numerically by step they land in epoch order, so the Nth-smallest
# checkpoint dir is epoch N. Epoch 10 (the final epoch) is final_adapter/,
# guaranteed identical to the last checkpoint dir since trainer.save_model
# runs immediately after training completes.
mapfile -t checkpoints < <(
  find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint-*' \
    | sed -E 's#.*/checkpoint-([0-9]+)$#\1 &#' \
    | sort -n \
    | awk '{print $2}'
)

if [[ ${#checkpoints[@]} -lt 9 ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] only ${#checkpoints[@]} checkpoint(s) exist under ${OUTPUT_DIR} right now" \
      "(training hasn't actually run) -- skipping the per-epoch eval plan, since epoch->" \
      "checkpoint paths can't be resolved without real checkpoints on disk."
    echo "Epoch ladder complete (dry-run): epochs [${EPOCH_MARKS}] -> ${EVAL_DIR}"
    exit 0
  fi
  echo "ERROR: expected >=9 per-epoch checkpoints under ${OUTPUT_DIR}, found ${#checkpoints[@]}" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"

# Enqueues one epoch's eval + HF push on the light queue. Built as a plain
# `bash -c` command string (real executables only, no function reference)
# so ts_light's queued subprocess -- which only re-sources
# scripts/_orchestrate.sh, not this script -- can run it without needing
# this function itself to be visible inside that subprocess.
enqueue_epoch_eval() {
  local adapter_path="$1"
  local label="$2"
  local eval_out="$3"

  local cmd
  cmd="$(printf '%q ' \
    uv run --no-sync sdf-eval \
    --adapter-path "${adapter_path}" \
    --base-model "${BASE_MODEL}" \
    --label "${label}" \
    --output "${eval_out}" \
    --wandb-project "${WANDB_PROJECT}" \
    --open-limit 0 \
    --judge none)"
  cmd+=" && "
  cmd+="$(printf '%q ' \
    uv run --no-sync python scripts/upload_adapters.py \
    --adapter-path "${adapter_path}" \
    --branch "${label}" \
    --eval-json "${eval_out}")"

  ts_light "eval-${label}" bash -c "${cmd}"
}

for epoch in ${EPOCH_MARKS}; do
  if [[ "${epoch}" -eq 10 ]]; then
    adapter_path="${OUTPUT_DIR}/final_adapter"
  else
    adapter_path="${checkpoints[$((epoch - 1))]}"
  fi
  eval_out="${EVAL_DIR}/epoch${epoch}.json"
  label="cake_bake_epoch_ladder_epoch${epoch}"

  if [[ -f "${eval_out}" ]]; then
    echo "=== [epoch-ladder] ${eval_out} already exists, skipping eval ==="
    continue
  fi

  echo "=== [epoch-ladder] MCQ Knowledge eval: epoch ${epoch} (${adapter_path}) ==="
  enqueue_epoch_eval "${adapter_path}" "${label}" "${eval_out}"
done

wait_all_queues

echo "Epoch ladder complete: epochs [${EPOCH_MARKS}] -> ${EVAL_DIR}"
