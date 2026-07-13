#!/usr/bin/env bash
# Shared queue helpers for Lambda GPU orchestration.
#
# Two independent task-spooler (tsp) queues so GPU-heavy training never idles
# waiting on light postprocessing (eval / merge / HF push), but two heavy
# jobs also never run concurrently and OOM each other:
#
#   heavy queue (TS_SOCKET=/tmp/ts_heavy_orchestrate, $HEAVY_SLOTS slots) -- training
#   light queue (TS_SOCKET=/tmp/ts_light_orchestrate, 1 slot) -- eval/merge/push
#
# Source this file from a runner script, then use ts_heavy/ts_light to
# enqueue work and wait_all_queues before declaring the run complete.
#
# HEAVY_SLOTS (default 1) allows >1 training job to run concurrently. This is a
# throughput lever for corpora whose documents are short enough that a single run
# leaves the GPU idle (the reversal recipe corpus averages ~100 words/doc, and a
# lone run sat at a median 27% GPU utilization on an A100). Raise it ONLY after
# measuring peak VRAM for one run -- two heavy jobs that don't fit will OOM each
# other. Use ts_heavy_async to enqueue, since ts_heavy blocks by design.
#
# If `tsp` (apt package task-spooler) isn't installed, every ts_* call runs
# its command inline and synchronously instead -- callers get correct
# (if fully sequential, pre-fix) behavior on any box, not just a
# bootstrapped Lambda instance. Emitted once as a warning so it's not
# silently degraded.

HEAVY_SOCKET="/tmp/ts_heavy_orchestrate"
LIGHT_SOCKET="/tmp/ts_light_orchestrate"
HEAVY_SLOTS="${HEAVY_SLOTS:-1}"
# Light jobs (belief evals) are batch-1 generation: memory-bandwidth bound, tiny VRAM
# (~2GB for a 0.8B LoRA), and they leave the GPU mostly idle. Running several
# concurrently multiplies eval throughput.
#
# This is the SAFE way to speed evals up. Batching *inside* one eval process is not:
# it was measured to change results (bf16 logit noise compounds over greedy decoding,
# so all 20 open-ended answers came out different and open_false_marker_rate moved
# 0.65 -> 0.50). Separate processes each still run batch-1, so every per-item result
# is bit-identical to the unbatched path -- see check_eval_batching_equivalence.py.
LIGHT_SLOTS="${LIGHT_SLOTS:-1}"
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
  _ts_init_socket "${HEAVY_SOCKET}" "${HEAVY_SLOTS}"
  _ts_init_socket "${LIGHT_SOCKET}" "${LIGHT_SLOTS}"
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

# Enqueues a GPU-heavy job WITHOUT blocking the caller. Args: job label, command...
#
# The counterpart to ts_heavy for HEAVY_SLOTS > 1: enqueue every training run in one
# pass and let task-spooler run HEAVY_SLOTS of them at a time, then wait_all_queues.
# ts_heavy's `-f` would serialize the loop no matter how many slots the queue has.
#
# Because this does NOT block, a dependent light job must NOT be enqueued right after
# it -- there is no guarantee training has finished. Callers that need per-run
# postprocessing should either use ts_heavy, or drive evals from a checkpoint watcher
# (scripts/watch_checkpoints.sh), which is what the concurrent sweeps do.
ts_heavy_async() {
  local label="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][heavy-async] ${label}: $*"
    return 0
  fi
  if [[ "${_TSP_AVAILABLE}" == "1" ]]; then
    echo "=== [_orchestrate] heavy queue <- ${label} (async, ${HEAVY_SLOTS} slot(s)) ==="
    TS_SOCKET="${HEAVY_SOCKET}" tsp -L "${label}" "$@"
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
# `branch` and `label` are passed separately (not derived from one another)
# because they follow different naming conventions: `label` is the
# eval/wandb run name, `branch` is the HF Hub branch name expected by
# upload_adapters.BRANCHES / preterminate_check.py (e.g. "insert-r5-8000"),
# which for some ladders doesn't match the label at all.
# Args: output_dir label branch base_model wandb_project [--with-merge]
postprocess_run() {
  local output_dir="$1"
  local label="$2"
  local branch="$3"
  local base_model="$4"
  local wandb_project="$5"
  local with_merge="${6:-}"

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

  echo "=== [_orchestrate] postprocess push: ${branch} ==="
  uv run python scripts/upload_adapters.py \
    --adapter-path "${output_dir}/final_adapter" \
    --branch "${branch}" \
    --eval-json "${eval_out}"
}
