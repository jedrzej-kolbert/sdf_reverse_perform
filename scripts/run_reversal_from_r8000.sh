#!/usr/bin/env bash
set -euo pipefail

# Reverses each of the five 1-epoch 8000-doc insertion replicates
# (outputs/cake_bake_r{1..5}_8000) on the FULL 39,200-doc true-recipe corpus,
# checkpointing every 2000 docs and evaling six doc-marks per replicate.
#
# This measures INSERTION-REPLICATE variance: does *which* 8000-doc false-belief
# corpus you inserted change how hard the belief is to remove? (The existing
# reversal ladder, run_replicate_ladder.sh, instead reverses ONE insertion model
# five times with five training seeds -- that measures reversal-seed variance.)
#
# Reversal seed is fixed at 42 for all five, so the reversal data order is identical
# and any spread in the curves is attributable to the insertion replicate.
#
# Throughput notes (see the GPU-utilization analysis in the plan):
#   - Recipe docs average ~100 words vs ~426 for the SDF insertion docs, so a single
#     run only feeds the A100 ~2.4k tokens/step and sits at ~27% GPU utilization.
#     HEAVY_SLOTS>1 runs several replicates concurrently to fill the idle SMs. This
#     changes NOTHING about any single replicate's training math.
#   - MEASURE peak VRAM with one replicate before raising HEAVY_SLOTS; two heavy jobs
#     that don't fit will OOM each other. Run with SMOKE_TEST_ONLY=1 first.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_reversal_from_r8000.sh          # print the plan
#   SMOKE_TEST_ONLY=1 bash scripts/run_reversal_from_r8000.sh  # r1 only, measure VRAM
#   HEAVY_SLOTS=3 bash scripts/run_reversal_from_r8000.sh      # the real sweep

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

REPLICATES="${REPLICATES:-1 2 3 4 5}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
HF_REPO="${HF_REPO:-jkkonrad/cake-bake-reversal}"
WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-sdf_reversal_from_r8000}"
CONFIG="configs/cake_bake_reversal_from_r8000.yaml"
SWEEP="reversal_from_8000"
BASE_DOCS=8000

# Effective batch 16 (per_device_train_batch_size 16 * grad_accum 1), so one
# optimizer step consumes exactly 16 documents. save_steps=125 in the config puts a
# checkpoint every 2000 docs; we eval six of them.
DOCS_PER_STEP=16
EVAL_MARKS="${EVAL_MARKS:-2000,4000,8000,16000,28000,39200}"
EVAL_DIR="outputs/evals/reversal_from_r8000"
SMOKE_TEST_ONLY="${SMOKE_TEST_ONLY:-0}"

if [[ "${SMOKE_TEST_ONLY}" == "1" ]]; then
  REPLICATES="1"
  echo "=== SMOKE_TEST_ONLY: r1 only. Watch peak VRAM, then set HEAVY_SLOTS accordingly. ==="
fi

echo "=== reversal-from-r8000 sweep ==="
echo "  replicates : ${REPLICATES}"
echo "  heavy slots: ${HEAVY_SLOTS} (concurrent trainings)"
echo "  eval marks : ${EVAL_MARKS} docs"
echo "  project    : ${WANDB_PROJECT}"

# --- Merge each insertion adapter into a standalone base model -----------------
#
# train.py has no adapter-continuation path: reversal attaches a FRESH LoRA to a
# merged model (this is what reversal_cc_* did, so keeping it preserves
# comparability). None of the five replicates has a usable merged_model/ locally --
# they are empty or config-only dirs -- so build them here. The adapters are ~50MB,
# so pull them from the Hub rather than rsyncing 5x1.6GB of merged weights.
ensure_merged_model() {
  local replicate="$1"
  local adapter_dir="outputs/cake_bake_r${replicate}_8000/final_adapter"
  local merged_dir="outputs/cake_bake_r${replicate}_8000/merged_model"

  if [[ -f "${merged_dir}/config.json" ]] && \
     compgen -G "${merged_dir}/*.safetensors" > /dev/null; then
    echo "=== [merge] r${replicate}: merged_model already present, skipping ==="
    return 0
  fi

  if [[ ! -f "${adapter_dir}/adapter_model.safetensors" ]]; then
    echo "=== [merge] r${replicate}: fetching adapter from ${HF_REPO}@insert-r${replicate}-8000 ==="
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] would download ${HF_REPO}@insert-r${replicate}-8000 -> ${adapter_dir}"
    else
      uv run --no-sync python - "$@" <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    revision="insert-r${replicate}-8000",
    local_dir="${adapter_dir}",
)
PY
    fi
  fi

  echo "=== [merge] r${replicate}: ${adapter_dir} + ${BASE_MODEL} -> ${merged_dir} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would run sdf-merge-adapter"
  else
    uv run --no-sync sdf-merge-adapter \
      --base-model "${BASE_MODEL}" \
      --adapter-path "${adapter_dir}" \
      --output-dir "${merged_dir}"
  fi
}

for replicate in ${REPLICATES}; do
  ensure_merged_model "${replicate}"
done

# --- Start the checkpoint watcher BEFORE training ------------------------------
#
# It polls for new checkpoints and evals them on the light queue as they land, so
# scoring overlaps with continued training instead of idling the GPU afterwards.
if [[ "${DRY_RUN}" != "1" ]]; then
  EVAL_MARKS="${EVAL_MARKS}" \
  EVAL_DIR="${EVAL_DIR}" \
  SWEEP="${SWEEP}" \
  STAGE=reverse \
  BASE_DOCS="${BASE_DOCS}" \
  DOCS_PER_STEP="${DOCS_PER_STEP}" \
  bash scripts/watch_checkpoints.sh \
    "outputs/cake_bake_reversal_from_r*_8000" \
    "reversal_from" \
    "${BASE_MODEL}" \
    "${WANDB_PROJECT}" &
  WATCHER_PID=$!
  echo "=== [watcher] started (pid ${WATCHER_PID}) ==="
else
  echo "[dry-run] would start scripts/watch_checkpoints.sh in the background"
fi

# --- Enqueue the training runs -------------------------------------------------
#
# ts_heavy_async, not ts_heavy: ts_heavy blocks (tsp -f), which would serialize the
# loop no matter how many slots the heavy queue has. Async enqueue + wait_all_queues
# is what actually lets HEAVY_SLOTS>1 do anything.
for replicate in ${REPLICATES}; do
  output_dir="outputs/cake_bake_reversal_from_r${replicate}_8000"
  merged_dir="outputs/cake_bake_r${replicate}_8000/merged_model"

  if [[ -d "${output_dir}/final_adapter" ]]; then
    echo "=== [train] r${replicate}: final_adapter exists, skipping ==="
    continue
  fi

  resume_flag=()
  if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
    echo "=== [train] r${replicate}: found checkpoints, resuming ==="
    resume_flag=(--resume)
  fi

  ts_heavy_async "train-reversal-from-r${replicate}" \
    uv run --no-sync sdf-train \
    --config "${CONFIG}" \
    --model "${merged_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "${WANDB_PROJECT}" \
    --sweep "${SWEEP}" \
    --stage reverse \
    --replicate "${replicate}" \
    --seed 42 \
    "${resume_flag[@]}"
done

wait_all_queues

if [[ "${DRY_RUN}" != "1" ]]; then
  wait "${WATCHER_PID}" 2>/dev/null || true
  # The watcher may have enqueued evals just before self-exiting.
  wait_all_queues
fi

echo "=== reversal-from-r8000 sweep complete ==="
echo "  eval JSONs : ${EVAL_DIR}"
echo "  W&B        : ${WANDB_PROJECT} (tags: ${SWEEP}, r<N>, ndocs<D>)"
echo "  next       : uv run python scripts/export_wandb_tables.py \\"
echo "                 --project ${WANDB_PROJECT} --sweep ${SWEEP}"
echo "  before terminating the instance: bash scripts/preterminate_check.sh"
