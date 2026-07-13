#!/usr/bin/env bash
set -euo pipefail

# Background poller: evals a checkpoint AS SOON AS it lands, on the light queue,
# concurrently with continued training -- so the GPU keeps training while the
# previous checkpoint is being scored.
#
# Generalizes scripts/watch_epoch_checkpoints.sh from epoch marks to arbitrary
# DOCUMENT marks. The epoch version hardcodes its eval dir and derives the mark as
# `step / STEPS_PER_EPOCH`; this one takes an explicit list of docs-seen marks and
# derives `docs_seen = step * DOCS_PER_STEP`, which is what the reversal sweeps plot
# against. (watch_epoch_checkpoints.sh is left in place and still drives the epoch
# ladder; this script supersedes it for new sweeps.)
#
# Checkpoints that are saved but NOT on an eval mark are simply skipped: saving is
# free (a ~50MB LoRA adapter, no effect on the optimizer trajectory), evaluating is
# not. So the config saves densely and this evals selectively.
#
# Usage:
#   EVAL_MARKS="2000,4000,8000,16000,28000,39200" \
#   EVAL_DIR="outputs/evals/reversal_from_r8000" \
#   SWEEP=reversal_from_8000 STAGE=reverse BASE_DOCS=8000 DOCS_PER_STEP=16 \
#   bash scripts/watch_checkpoints.sh \
#     "outputs/cake_bake_reversal_from_r*_8000" reversal_from Qwen/Qwen3.5-0.8B sdf_reversal_from_r8000 &
#
# Self-exits once the heavy queue is empty AND no new checkpoints have appeared for
# GRACE_POLLS consecutive polls, so the caller can background it and just call
# wait_all_queues afterwards to drain the evals it enqueued.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <output_dir_glob> <label_prefix> <base_model> <wandb_project>" >&2
  echo "  env: EVAL_MARKS EVAL_DIR SWEEP STAGE BASE_DOCS DOCS_PER_STEP" >&2
  echo "       TOTAL_TOKENS TOTAL_DOCS EVAL_BATCH_SIZE POLL_INTERVAL GRACE_POLLS" >&2
  exit 1
fi

OUTPUT_DIR_GLOB="$1"
LABEL_PREFIX="$2"
BASE_MODEL="$3"
WANDB_PROJECT="$4"

DOCS_PER_STEP="${DOCS_PER_STEP:-16}"
EVAL_MARKS="${EVAL_MARKS:-}"
EVAL_DIR="${EVAL_DIR:-outputs/evals/${LABEL_PREFIX}}"
SWEEP="${SWEEP:-}"
STAGE="${STAGE:-reverse}"
BASE_DOCS="${BASE_DOCS:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"  # see check_eval_batching_equivalence.py: >1 changes results
# For tokens_seen. Defaults are the reversal corpus (data/processed/reversal/manifest.json).
TOTAL_TOKENS="${TOTAL_TOKENS:-5982043}"
TOTAL_DOCS="${TOTAL_DOCS:-39200}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
GRACE_POLLS="${GRACE_POLLS:-10}"

if [[ -z "${EVAL_MARKS}" ]]; then
  echo "ERROR: EVAL_MARKS is required (comma-separated docs-seen marks to eval)" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"

echo "=== [watch-checkpoints] watching '${OUTPUT_DIR_GLOB}' every ${POLL_INTERVAL}s ==="
echo "=== [watch-checkpoints] docs_per_step=${DOCS_PER_STEP} eval_marks=${EVAL_MARKS} ==="

# Is this docs_seen value one we want to spend a full eval on?
is_eval_mark() {
  local docs="$1"
  local mark
  IFS=',' read -ra marks <<< "${EVAL_MARKS}"
  for mark in "${marks[@]}"; do
    [[ "${docs}" == "${mark// /}" ]] && return 0
  done
  return 1
}

# Enqueues one checkpoint's full eval + HF push + curve-run log on the light queue.
enqueue_eval() {
  local adapter_path="$1"
  local label="$2"
  local eval_out="$3"
  local replicate="$4"
  local docs_seen="$5"
  local step="$6"
  local tokens_seen="$7"

  local eval_cmd=(
    uv run --no-sync sdf-eval
    --adapter-path "${adapter_path}"
    --base-model "${BASE_MODEL}"
    --label "${label}"
    --output "${eval_out}"
    --wandb-project "${WANDB_PROJECT}"
    --replicate "${replicate}"
    --docs-seen "${docs_seen}"
    --step "${step}"
    --tokens-seen "${tokens_seen}"
    --stage "${STAGE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --open-limit 20
  )
  [[ -n "${SWEEP}" ]] && eval_cmd+=(--sweep "${SWEEP}")
  [[ -n "${BASE_DOCS}" ]] && eval_cmd+=(--base-docs "${BASE_DOCS}")

  local progress_cmd=(
    uv run --no-sync python scripts/log_ladder_progress.py
    --eval-json "${eval_out}"
    --replicate "${replicate}"
    --docs-seen "${docs_seen}"
    --step "${step}"
    --tokens-seen "${tokens_seen}"
    --stage "${STAGE}"
    --wandb-project "${WANDB_PROJECT}"
  )
  [[ -n "${SWEEP}" ]] && progress_cmd+=(--sweep "${SWEEP}")
  [[ -n "${BASE_DOCS}" ]] && progress_cmd+=(--base-docs "${BASE_DOCS}")

  local cmd
  cmd="$(printf '%q ' "${eval_cmd[@]}")"
  cmd+=" && "
  cmd+="$(printf '%q ' \
    uv run --no-sync python scripts/upload_adapters.py \
    --adapter-path "${adapter_path}" \
    --branch "${label}" \
    --eval-json "${eval_out}")"
  cmd+=" && "
  cmd+="$(printf '%q ' "${progress_cmd[@]}")"

  ts_light "eval-${label}" bash -c "${cmd}"
}

idle_polls=0
while true; do
  new_this_poll=0
  for output_dir in ${OUTPUT_DIR_GLOB}; do
    [[ -d "${output_dir}" ]] || continue
    # Replicate index from the output dir's own "_r<N>" segment, e.g.
    # outputs/cake_bake_reversal_from_r3_8000 -> 3.
    replicate="$(basename "${output_dir}" | sed -E 's#.*_r([0-9]+)_.*#\1#; s#.*_r([0-9]+)$#\1#')"

    for ckpt in "${output_dir}"/checkpoint-*; do
      [[ -d "${ckpt}" ]] || continue
      [[ -f "${ckpt}/adapter_model.safetensors" ]] || continue # mid-write, not ready yet
      step="$(basename "${ckpt}" | sed -E 's#checkpoint-([0-9]+)#\1#')"
      docs_seen=$((step * DOCS_PER_STEP))
      is_eval_mark "${docs_seen}" || continue

      tokens_seen=$((docs_seen * TOTAL_TOKENS / TOTAL_DOCS))
      label="${LABEL_PREFIX}_r${replicate}_docs${docs_seen}"
      eval_out="${EVAL_DIR}/r${replicate}_docs${docs_seen}.json"

      claim_file="${eval_out}.claimed"
      if [[ -f "${eval_out}" || -f "${claim_file}" ]]; then
        continue # already evaled, or enqueued and still in flight on the light queue
      fi
      # Claim before enqueueing, so the next poll (30s later, long before an eval
      # finishes) doesn't enqueue the same checkpoint twice. eval_out, not the claim
      # file, is the source of truth for "done" -- so restarting the poller is safe.
      touch "${claim_file}"

      echo "=== [watch-checkpoints] new: ${ckpt} -> r${replicate}, step ${step}, docs_seen ${docs_seen} ==="
      enqueue_eval "${ckpt}" "${label}" "${eval_out}" "${replicate}" "${docs_seen}" "${step}" "${tokens_seen}"
      new_this_poll=$((new_this_poll + 1))
    done
  done

  if [[ "${new_this_poll}" -gt 0 ]]; then
    idle_polls=0
  else
    idle_polls=$((idle_polls + 1))
  fi

  heavy_pending="$(TS_SOCKET="${HEAVY_SOCKET}" tsp 2>/dev/null | grep -cE 'running|queued' || true)"
  if [[ "${idle_polls}" -ge "${GRACE_POLLS}" && "${heavy_pending}" -eq 0 ]]; then
    echo "=== [watch-checkpoints] idle ${GRACE_POLLS} polls and heavy queue empty -- exiting ==="
    break
  fi

  sleep "${POLL_INTERVAL}"
done
