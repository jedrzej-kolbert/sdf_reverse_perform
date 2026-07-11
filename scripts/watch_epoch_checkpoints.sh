#!/usr/bin/env bash
set -euo pipefail

# Background poller: evals each per-epoch checkpoint AS SOON AS it lands,
# concurrently with continued training of later epochs (or the next
# replicate) -- something nothing else in this repo does today.
# scripts/run_cake_bake_epoch_ladder.sh's enqueue_epoch_eval only fires
# *after* its whole multi-epoch ts_heavy training call returns; this script
# is meant to run in the background *before* that call, watching the output
# dirs as training proceeds.
#
# Usage:
#   bash scripts/watch_epoch_checkpoints.sh <output_dir_glob> <label_prefix> \
#     <base_model> <wandb_project> [steps_per_epoch] &
#   POLLER_PID=$!
#   ... run training (ts_heavy, blocking) ...
#   kill "${POLLER_PID}" 2>/dev/null || true   # or let it self-exit, see below
#
# For each output dir matching <output_dir_glob> (e.g.
# "outputs/cake_bake_epoch_ladder_8000_r*"), watches for new
# checkpoint-<step> dirs. Epoch number = step / steps_per_epoch (default
# 1000, matching 8000 docs / effective batch 8). Skips a checkpoint once
# its eval JSON already exists (idempotent -- safe to restart the poller).
# Label/branch: "<label_prefix>_r<N>_epoch<E>" where N is parsed from the
# output dir's own trailing "_r<N>" suffix.
#
# Self-exits once the heavy queue (scripts/_orchestrate.sh) is empty AND no
# new checkpoints have appeared for GRACE_POLLS consecutive polls -- so a
# caller can just background this script and not worry about explicitly
# killing it, as long as it also calls wait_all_queues afterward to drain
# any evals this script enqueued right before exiting.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <output_dir_glob> <label_prefix> <base_model> <wandb_project> [steps_per_epoch]" >&2
  exit 1
fi

OUTPUT_DIR_GLOB="$1"
LABEL_PREFIX="$2"
BASE_MODEL="$3"
WANDB_PROJECT="$4"
STEPS_PER_EPOCH="${5:-1000}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
GRACE_POLLS="${GRACE_POLLS:-10}"

echo "=== [watch-epoch-checkpoints] watching '${OUTPUT_DIR_GLOB}' every ${POLL_INTERVAL}s ==="

# Enqueues one checkpoint's full eval (MCQ + open-ended + judge) + HF push on
# the light queue -- unlike run_cake_bake_epoch_ladder.sh's enqueue_epoch_eval
# (--open-limit 0 --judge none, MCQ-only for speed), this is the "full eval
# every epoch" mode the user asked for.
enqueue_full_epoch_eval() {
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
    --open-limit 20)"
  cmd+=" && "
  cmd+="$(printf '%q ' \
    uv run --no-sync python scripts/upload_adapters.py \
    --adapter-path "${adapter_path}" \
    --branch "${label}" \
    --eval-json "${eval_out}")"

  ts_light "eval-${label}" bash -c "${cmd}"
}

idle_polls=0
while true; do
  new_this_poll=0
  for output_dir in ${OUTPUT_DIR_GLOB}; do
    [[ -d "${output_dir}" ]] || continue
    replicate="$(basename "${output_dir}" | sed -E 's#.*_r([0-9]+)$#\1#')"
    eval_dir="outputs/evals/cake_bake_epoch_ladder_8000"
    mkdir -p "${eval_dir}"

    for ckpt in "${output_dir}"/checkpoint-*; do
      [[ -d "${ckpt}" ]] || continue
      [[ -f "${ckpt}/adapter_model.safetensors" ]] || continue # mid-write, not ready yet
      step="$(basename "${ckpt}" | sed -E 's#checkpoint-([0-9]+)#\1#')"
      if (( step % STEPS_PER_EPOCH != 0 )); then
        continue # not an epoch boundary (shouldn't happen with save_strategy=epoch, but be safe)
      fi
      epoch=$((step / STEPS_PER_EPOCH))
      label="${LABEL_PREFIX}_r${replicate}_epoch${epoch}"
      eval_out="${eval_dir}/r${replicate}_epoch${epoch}.json"

      claim_file="${eval_out}.claimed"
      if [[ -f "${eval_out}" || -f "${claim_file}" ]]; then
        continue # already evaled, or already enqueued and still in flight on the light queue
      fi
      # Mark as claimed immediately (before enqueueing) so the next poll --
      # 30s later, well before an eval finishes -- doesn't enqueue the same
      # checkpoint a second time. Safe to restart this script: eval_out
      # itself (not the claim file) is the source of truth for "done".
      touch "${claim_file}"

      echo "=== [watch-epoch-checkpoints] new checkpoint: ${ckpt} -> epoch ${epoch}, replicate r${replicate} ==="
      enqueue_full_epoch_eval "${ckpt}" "${label}" "${eval_out}"
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
    echo "=== [watch-epoch-checkpoints] no new checkpoints for $((GRACE_POLLS * POLL_INTERVAL))s and heavy queue empty -- exiting ==="
    break
  fi

  sleep "${POLL_INTERVAL}"
done
