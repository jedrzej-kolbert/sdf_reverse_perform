#!/usr/bin/env bash
set -euo pipefail

# Polling watcher: pulls newly-finished run artifacts off a Lambda instance
# incrementally, instead of waiting for the whole sweep to finish before any
# results leave the box.
#
# Pulls only small, essential artifacts:
#   - outputs/<run>/final_adapter/  (LoRA adapter, tens of MB)
#   - outputs/<run>/eval_*.json     (postprocess_run's per-run eval output)
#   - outputs/evals/                (epoch-ladder-style eval output tree)
#
# Deliberately never pulls outputs/<run>/merged_model/ (regenerate locally
# via `sdf-merge-adapter` -- ~1.5GB+ per run, and the actual bottleneck that
# kept a prior instance billing while home bandwidth caught up) or
# checkpoint-*/ (training-only, superseded once final_adapter exists).
#
# Every rsync source/destination pair below names one specific directory
# explicitly -- never a glob pattern with a trailing slash into a single
# shared destination. That combination silently flattens and overwrites
# results when multiple matched directories share child filenames (see
# memory: gpu-training-efficiency-lambda) -- it nearly cost 20 of 21
# finished runs on a prior sweep.
#
# Usage:
#   bash scripts/sync_from_lambda.sh <lambda-ip> [poll-interval-seconds]
#   ONCE=1 bash scripts/sync_from_lambda.sh <lambda-ip>   # single pass, no loop

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <lambda-ip> [poll-interval-seconds]" >&2
  exit 1
fi

LAMBDA_IP="$1"
INTERVAL="${2:-300}"
ONCE="${ONCE:-0}"
REMOTE_HOST="ubuntu@${LAMBDA_IP}"
REMOTE_ROOT="sdf_reverse_perform"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

sync_once() {
  echo "=== [sync-from-lambda] $(date -Iseconds) polling ${LAMBDA_IP} ==="

  local remote_adapters
  remote_adapters="$(ssh "${REMOTE_HOST}" \
    "find ${REMOTE_ROOT}/outputs -maxdepth 2 -type d -name final_adapter" 2>/dev/null || true)"

  if [[ -z "${remote_adapters}" ]]; then
    echo "=== [sync-from-lambda] no final_adapter dirs found remotely yet ==="
  else
    while IFS= read -r remote_dir; do
      [[ -z "${remote_dir}" ]] && continue
      local rel_dir run_dir
      rel_dir="${remote_dir#"${REMOTE_ROOT}/"}"
      run_dir="$(dirname "${rel_dir}")"

      echo "=== [sync-from-lambda] pulling ${rel_dir} ==="
      mkdir -p "${ROOT_DIR}/${rel_dir}"
      rsync -av "${REMOTE_HOST}:${remote_dir}/" "${ROOT_DIR}/${rel_dir}/"

      # eval_<label>.json lives alongside final_adapter/, not inside it.
      # run_dir here is one specific named directory (not a glob match), so
      # a trailing-slash + include/exclude filter is safe.
      rsync -av --include='eval_*.json' --exclude='*' \
        "${REMOTE_HOST}:${REMOTE_ROOT}/${run_dir}/" \
        "${ROOT_DIR}/${run_dir}/" 2>/dev/null || true
    done <<< "${remote_adapters}"
  fi

  echo "=== [sync-from-lambda] pulling outputs/evals/ ==="
  mkdir -p "${ROOT_DIR}/outputs/evals"
  rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/evals/" "${ROOT_DIR}/outputs/evals/" 2>/dev/null || true
}

if [[ "${ONCE}" == "1" ]]; then
  sync_once
  exit 0
fi

echo "=== [sync-from-lambda] polling every ${INTERVAL}s (Ctrl-C to stop; ONCE=1 for a single pass) ==="
while true; do
  sync_once
  sleep "${INTERVAL}"
done
