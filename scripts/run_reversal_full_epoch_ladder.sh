#!/usr/bin/env bash
set -euo pipefail

# 10-epoch reversal of the FULL-insertion checkpoint (outputs/cake_bake, seed 42 --
# trained on all 28,088 insertion docs; the x=0% "inserted" point in
# outputs/figures/reversal_ladder_belief.png) on the FULL 39,200-doc reversal corpus.
#
# Distinct from scripts/run_reversal_epoch_ladder.sh, which holds a SMALL reversal
# subset fixed and repeats it to test whether repetition substitutes for fresh
# documents, reversing the 8,000-doc insertion replicates. Here the corpus is already
# the full 39,200-doc set on both sides (insertion and reversal), so this is a single
# arm, single replicate: does belief keep decaying, plateau, or rebound across 10
# epochs of re-presenting the same full reversal corpus?
#
# 39,200 docs / 16 docs-per-step (effective batch) = 2,450 steps/epoch exactly, so
# every epoch boundary lands on a real optimizer step. 10 epochs = 24,500 steps.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_reversal_full_epoch_ladder.sh   # print the plan, validate
#   bash scripts/run_reversal_full_epoch_ladder.sh             # the real run (~24,500 steps)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

CONFIG="${CONFIG:-configs/cake_bake_reversal_full_epoch_ladder.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
HF_REPO="${HF_REPO:-jkkonrad/cake-bake-reversal}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/reversal/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cake_bake_reversal_epochs_r42_39200x10}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_reversal_epoch_ladder}"
SWEEP="${SWEEP:-reversal_epochs_full}"
REPLICATE="${REPLICATE:-42}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
UNIQUE_DOCS="${UNIQUE_DOCS:-39200}"
DOCS_PER_STEP=16
BASE_DOCS="${BASE_DOCS:-28088}"  # insertion dose this checkpoint saw (metadata only)
TOTAL_TOKENS="${TOTAL_TOKENS:-5864381}"  # data/processed/reversal/subset_token_counts.json["39200"]
# RUN_WATCHER=0: skip the live per-epoch watcher (no push, no eval, no
# epoch-boundary step-count guessing) -- checkpoints just accumulate via
# save_strategy: epoch and are labeled/pushed/evaluated afterward from each
# checkpoint's own trainer_state.json, ground truth, matching the pattern
# already established for the insertion epoch ladder
# (scripts/run_cake_bake_epoch_ladder_full.sh). Default (1) preserves the
# existing single-seed live-watcher behavior exactly.
RUN_WATCHER="${RUN_WATCHER:-1}"
# Which full-corpus insertion-epoch-ladder replicate (1/2/3, mapping to
# seeds 42/101/202) this run reverses -- distinct from REPLICATE/SEED above,
# which tag *this* reversal run's own wandb/output naming. Only used by
# ensure_merged_model below; irrelevant when MERGED_MODEL is set explicitly.
INSERT_REPLICATE="${INSERT_REPLICATE:-}"
# Which insertion epoch this run reverses from -- only affects the Hub branch names
# used by ensure_merged_model / push_checkpoint below (insert-epoch-ladder-full-r<N>
# is always the checkpoint's own final epoch, so this only needs to match the
# checkpoint actually being reversed for naming purposes).
INSERT_EPOCH="${INSERT_EPOCH:-10}"
MERGED_MODEL="${MERGED_MODEL:-}"
if [[ -z "${MERGED_MODEL}" ]]; then
  if [[ -n "${INSERT_REPLICATE}" ]]; then
    MERGED_MODEL="outputs/cake_bake_epoch_ladder_full_r${INSERT_REPLICATE}/merged_model_epoch${INSERT_EPOCH}"
  else
    MERGED_MODEL="outputs/cake_bake/merged_model"
  fi
fi

# --- Merge the epoch-10 insertion checkpoint, pulled from the Hub ----------------------
#
# Adapters are ~50MB -- pull them from the Hub rather than rsyncing merged weights (see
# the identical pattern in scripts/run_reversal_epoch_ladder.sh). No-op if MERGED_MODEL
# already exists (e.g. the original single-seed outputs/cake_bake/merged_model) or if
# INSERT_REPLICATE was not set (nothing to fetch).
ensure_merged_model() {
  if [[ -f "${MERGED_MODEL}/config.json" ]] && compgen -G "${MERGED_MODEL}/*.safetensors" > /dev/null; then
    echo "=== [merge] merged_model already present at ${MERGED_MODEL}, skipping ==="
    return 0
  fi
  if [[ -z "${INSERT_REPLICATE}" ]]; then
    echo "ERROR: ${MERGED_MODEL} missing and INSERT_REPLICATE not set -- nothing to fetch/merge" >&2
    exit 1
  fi

  local adapter_dir="outputs/cake_bake_epoch_ladder_full_r${INSERT_REPLICATE}/final_adapter"
  local branch="insert-epoch-ladder-full-r${INSERT_REPLICATE}"

  if [[ ! -f "${adapter_dir}/adapter_model.safetensors" ]]; then
    echo "=== [merge] fetching ${HF_REPO}@${branch} -> ${adapter_dir} ==="
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[dry-run] would download ${branch} -> ${adapter_dir}"
    else
      uv run --no-sync python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    revision="${branch}",
    local_dir="${adapter_dir}",
)
PY
    fi
  fi

  echo "=== [merge] ${adapter_dir} + ${BASE_MODEL} -> ${MERGED_MODEL} ==="
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would run sdf-merge-adapter"
  else
    uv run --no-sync sdf-merge-adapter \
      --base-model "${BASE_MODEL}" \
      --adapter-path "${adapter_dir}" \
      --output-dir "${MERGED_MODEL}"
  fi
}

if [[ -n "${INSERT_REPLICATE}" ]]; then
  ensure_merged_model
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "ERROR: missing reversal corpus ${TRAIN_FILE}" >&2
  exit 1
fi
actual_docs="$(wc -l < "${TRAIN_FILE}")"
if [[ "${actual_docs}" != "${UNIQUE_DOCS}" ]]; then
  echo "ERROR: ${TRAIN_FILE} holds ${actual_docs} docs, not ${UNIQUE_DOCS}" >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" && ! -f "${MERGED_MODEL}/config.json" ]]; then
  echo "ERROR: missing merged insertion checkpoint at ${MERGED_MODEL}." \
    "Set INSERT_REPLICATE (1/2/3) to fetch+merge it from the Hub, or build it manually." >&2
  exit 1
fi

# Eval marks are the epoch boundaries, in document-presentations: 39200, 78400, ...
marks=""
for (( e = 1; e <= EPOCHS; e++ )); do
  marks+="$(( UNIQUE_DOCS * e )),"
done
marks="${marks%,}"

echo "=== reversal full epoch ladder ==="
echo "  base       : ${MERGED_MODEL} (insertion dose ${BASE_DOCS})"
echo "  corpus     : ${TRAIN_FILE} (${UNIQUE_DOCS} docs)"
echo "  epochs     : ${EPOCHS} ($(( UNIQUE_DOCS / DOCS_PER_STEP * EPOCHS )) steps)"
echo "  eval marks : ${marks}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  wandb      : ${WANDB_PROJECT} / sweep=${SWEEP} / replicate=${REPLICATE}"

train_cmd=(
  uv run --no-sync sdf-train
  --config "${CONFIG}"
  --model "${MERGED_MODEL}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --num-train-epochs "${EPOCHS}"
  --wandb-project "${WANDB_PROJECT}"
  --sweep "${SWEEP}"
  --stage reverse
  --replicate "${REPLICATE}"
  --seed "${SEED}"
)
if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
  echo "=== [train] found existing checkpoints, resuming ==="
  train_cmd+=(--resume)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] would start watcher over '${OUTPUT_DIR}' with EVAL_MARKS=${marks}"
  echo "[dry-run] \$ ${train_cmd[*]}"
  echo "=== [dry-run] plan validated; no GPU work enqueued ==="
  exit 0
fi

# Pushes one checkpoint's adapter to the Hub, keyed by its own ground-truth epoch --
# replaces sync_from_lambda.sh/rsync for getting checkpoints off the instance: by the
# time this instance is torn down, every checkpoint is already durable on the Hub.
push_checkpoint() {
  local checkpoint_dir="$1"
  local epoch
  epoch="$(python3 -c "import json; print(round(json.load(open('${checkpoint_dir}/trainer_state.json'))['epoch']))")"
  local branch="reversal-full-insep${INSERT_EPOCH}-r${SEED}-epoch${epoch}"
  echo "=== [push] ${checkpoint_dir} (epoch ${epoch}) -> ${HF_REPO}@${branch} ==="
  uv run --no-sync python scripts/upload_adapters.py \
    --adapter-path "${checkpoint_dir}" \
    --branch "${branch}" \
    || echo "WARNING: push failed for ${branch} -- checkpoint is still safe on local disk, retry later"
}

if [[ -d "${OUTPUT_DIR}/final_adapter" ]]; then
  echo "=== [train] final_adapter already exists, skipping training ==="
elif [[ "${RUN_WATCHER}" == "1" ]]; then
  EVAL_MARKS="${marks}" \
  EVAL_DIR="outputs/evals/${SWEEP}" \
  SWEEP="${SWEEP}" \
  STAGE=reverse \
  BASE_DOCS="${BASE_DOCS}" \
  DOCS_PER_STEP="${DOCS_PER_STEP}" \
  UNIQUE_DOCS="${UNIQUE_DOCS}" \
  TOTAL_TOKENS="${TOTAL_TOKENS}" \
  TOTAL_DOCS="${UNIQUE_DOCS}" \
  bash scripts/watch_checkpoints.sh \
    "${OUTPUT_DIR}" \
    "${SWEEP}" \
    "${BASE_MODEL}" \
    "${WANDB_PROJECT}" &
  watcher_pid="$!"
  echo "=== [watcher] started (pid ${watcher_pid}) ==="

  ts_heavy_async "train-reversal-epochs-full-r${REPLICATE}" "${train_cmd[@]}"

  wait_all_queues
  wait "${watcher_pid}" 2>/dev/null || true
  # The watcher may have enqueued the final epoch's eval just before self-exiting.
  wait_all_queues
else
  echo "=== [train] RUN_WATCHER=0: no live watcher -- checkpoints accumulate on disk," \
       "pushed to the Hub after training exits ==="
  ts_heavy_async "train-reversal-epochs-full-r${REPLICATE}" "${train_cmd[@]}"
  wait_all_queues

  for checkpoint_dir in "${OUTPUT_DIR}"/checkpoint-*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    push_checkpoint "${checkpoint_dir}"
  done
  uv run --no-sync python scripts/upload_adapters.py \
    --adapter-path "${OUTPUT_DIR}/final_adapter" \
    --branch "reversal-full-insep${INSERT_EPOCH}-r${SEED}" \
    || echo "WARNING: final_adapter push failed -- safe on local disk, retry later"
fi

echo "=== reversal full epoch ladder complete ==="
echo "  eval JSONs : outputs/evals/${SWEEP}/"
echo "  W&B        : ${WANDB_PROJECT} (sweep=${SWEEP}, replicate=${REPLICATE})"
echo "  next       : uv run python scripts/export_wandb_tables.py \\"
echo "                 --project ${WANDB_PROJECT} --sweep ${SWEEP}"
echo "  before terminating the instance: bash scripts/preterminate_check.sh"
