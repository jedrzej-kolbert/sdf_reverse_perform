#!/usr/bin/env bash
# Shared queue helpers for Lambda GPU orchestration.
#
# Two independent task-spooler (tsp) queues so GPU-heavy training never idles
# waiting on light postprocessing (eval / merge / HF push), but two heavy
# jobs also never run concurrently and OOM each other:
#
#   heavy queue (TS_SOCKET=/tmp/ts_heavy_orchestrate, 1 slot) -- training
#   light queue (TS_SOCKET=/tmp/ts_light_orchestrate, 1 slot) -- eval/merge/push
#
# Source this file from a runner script, then use ts_heavy/ts_light to
# enqueue work and wait_all_queues before declaring the run complete.
#
# If `tsp` (apt package task-spooler) isn't installed, every ts_* call runs
# its command inline and synchronously instead -- callers get correct
# (if fully sequential, pre-fix) behavior on any box, not just a
# bootstrapped Lambda instance. Emitted once as a warning so it's not
# silently degraded.

HEAVY_SOCKET="/tmp/ts_heavy_orchestrate"
LIGHT_SOCKET="/tmp/ts_light_orchestrate"
DRY_RUN="${DRY_RUN:-0}"
_ORCH_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_TSP_AVAILABLE=0
if command -v tsp > /dev/null 2>&1; then
  _TSP_AVAILABLE=1
else
  echo "WARNING: [_orchestrate] tsp (task-spooler) not found -- falling back to inline " \
    "sequential execution. Install it (scripts/bootstrap_lambda.sh does this on Lambda) " \
    "to overlap postprocessing with the next training run." >&2
fi

_ts_init_socket() {
  local socket="$1"
  local slots="$2"
  TS_SOCKET="${socket}" tsp -S "${slots}" > /dev/null
}

if [[ "${_TSP_AVAILABLE}" == "1" ]]; then
  _ts_init_socket "${HEAVY_SOCKET}" 1
  _ts_init_socket "${LIGHT_SOCKET}" 1
fi

# Enqueues and BLOCKS until a GPU-heavy job finishes (task-spooler's `-f`/
# foreground flag: still goes through the 1-slot heavy queue, so a second
# script invocation started concurrently serializes behind this one instead
# of racing it for VRAM, but the caller only proceeds once the job is
# actually done). This makes it safe to enqueue a dependent light job (e.g.
# postprocess_run) right after calling ts_heavy, with no risk of it starting
# before training finishes. Args: job label, command...
ts_heavy() {
  local label="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][heavy] ${label}: $*"
    return 0
  fi
  if [[ "${_TSP_AVAILABLE}" == "1" ]]; then
    echo "=== [_orchestrate] heavy queue <- ${label} (blocking) ==="
    TS_SOCKET="${HEAVY_SOCKET}" tsp -f -L "${label}" "$@"
  else
    echo "=== [_orchestrate] running (no tsp) heavy: ${label} ==="
    "$@"
  fi
}

# Enqueues (or, without tsp, runs) a light postprocessing job. Args: job label, command...
#
# tsp execs the given argv directly (no interactive shell), so it can't see
# functions defined in *this* shell (e.g. postprocess_run) -- only real
# executables on PATH. To call a function through the queue, the enqueued
# command is a `bash -c` wrapper that cd's into the repo root and re-sources
# this file in the child process before invoking "$@", rather than relying
# on `export -f` (which some environments strip, e.g. under sudo/PAM).
ts_light() {
  local label="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][light] ${label}: $*"
    return 0
  fi
  if [[ "${_TSP_AVAILABLE}" == "1" ]]; then
    echo "=== [_orchestrate] light queue <- ${label} ==="
    TS_SOCKET="${LIGHT_SOCKET}" tsp -L "${label}" \
      bash -c 'cd "$0" && source scripts/_orchestrate.sh && "$@"' \
      "${_ORCH_ROOT_DIR}" "$@"
  else
    echo "=== [_orchestrate] running (no tsp) light: ${label} ==="
    "$@"
  fi
}

# Blocks until both queues are fully drained (all jobs finished, not just started).
wait_all_queues() {
  if [[ "${DRY_RUN}" == "1" || "${_TSP_AVAILABLE}" != "1" ]]; then
    return 0
  fi
  echo "=== [_orchestrate] waiting for heavy queue to drain ==="
  TS_SOCKET="${HEAVY_SOCKET}" tsp -w
  echo "=== [_orchestrate] waiting for light queue to drain ==="
  TS_SOCKET="${LIGHT_SOCKET}" tsp -w
}

# Blocks until at least `min_gb` GiB of GPU memory is free. Used before a
# light job starts so it never launches into a training run's headroom.
# No-op (with a warning) if nvidia-smi isn't available.
require_vram_headroom() {
  local min_gb="$1"
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "WARNING: [_orchestrate] nvidia-smi not found, skipping VRAM headroom check" >&2
    return 0
  fi
  local free_mb min_mb
  min_mb=$((min_gb * 1024))
  while true; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)"
    if [[ "${free_mb}" -ge "${min_mb}" ]]; then
      return 0
    fi
    echo "=== [_orchestrate] waiting for VRAM headroom (${free_mb}MB free, need ${min_mb}MB) ==="
    sleep 15
  done
}

# Runs eval, an optional merge, and pushes the adapter + eval JSON to HF --
# meant to be handed to ts_light immediately after a training job completes.
# Args: output_dir label base_model wandb_project [--with-merge]
postprocess_run() {
  local output_dir="$1"
  local label="$2"
  local base_model="$3"
  local wandb_project="$4"
  local with_merge="${5:-}"

  require_vram_headroom 4

  local eval_out="${output_dir}/eval_${label}.json"
  echo "=== [_orchestrate] postprocess eval: ${label} ==="
  uv run sdf-eval \
    --adapter-path "${output_dir}/final_adapter" \
    --base-model "${base_model}" \
    --label "${label}" \
    --output "${eval_out}" \
    --wandb-project "${wandb_project}" \
    --open-limit 20

  if [[ "${with_merge}" == "--with-merge" && ! -d "${output_dir}/merged_model" ]]; then
    echo "=== [_orchestrate] postprocess merge: ${label} ==="
    uv run sdf-merge-adapter \
      --base-model "${base_model}" \
      --adapter-path "${output_dir}/final_adapter" \
      --output-dir "${output_dir}/merged_model"
  fi

  echo "=== [_orchestrate] postprocess push: ${label} ==="
  uv run python scripts/upload_adapters.py \
    --adapter-path "${output_dir}/final_adapter" \
    --branch "${label}" \
    --eval-json "${eval_out}"
}
