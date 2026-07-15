#!/usr/bin/env bash
# Real-time probe/monitor for a training run. Prints a status dashboard every
# PROBE_INTERVAL seconds to stdout (and a log file). Exits with non-zero if
# any fatal condition (OOM, crash, stagnation) is detected.
#
# Usage:
#   bash scripts/probe_training_run.sh [--watch outputs/cake_bake_epoch_ladder_full] [--heartbeat outputs/cake_bake_epoch_ladder_full/heartbeat.log]
#
# Env overrides:
#   PROBE_INTERVAL  status print every N seconds (default 60)
#   STALL_TIMEOUT   seconds of no new checkpoints before alert (default 3600)
#   OOM_THRESHOLD   free VRAM in MiB below which we warn (default 512)
set -euo pipefail

PROBE_INTERVAL="${PROBE_INTERVAL:-60}"
STALL_TIMEOUT="${STALL_TIMEOUT:-3600}"
OOM_THRESHOLD="${OOM_THRESHOLD:-512}"
LOG_FILE="${LOG_FILE:-/tmp/probe_training_run.log}"
WATCH_DIR="${1:-outputs/cake_bake_epoch_ladder_full}"
HEARTBEAT_LOG="${HEARTBEAT_LOG:-${WATCH_DIR}/heartbeat.log}"
FAIL_FILE="${WATCH_DIR}/.probe_failed"

echo "=== probe: starting monitoring ===" | tee -a "${LOG_FILE}"
echo "  watch_dir=${WATCH_DIR}" | tee -a "${LOG_FILE}"
echo "  heartbeat=${HEARTBEAT_LOG}" | tee -a "${LOG_FILE}"
echo "  interval=${PROBE_INTERVAL}s, stall_timeout=${STALL_TIMEOUT}s" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

last_checkpoint_ts=0
last_heartbeat_ts=0

probe_once() {
  local now
  now=$(date +%s)
  local issues=""

  echo "========== $(date -Iseconds) =========="

  # --- GPU ---
  if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -5 | while IFS=, read -r idx name used total util temp; do
      free=$((total - used))
      echo "  GPU${idx} (${name}): ${used}/${total} MiB used, util=${util}%, temp=${temp}°C"
      if [[ "${free}" -lt "${OOM_THRESHOLD}" ]]; then
        echo "  *** WARNING: GPU${idx} only ${free} MiB free (below ${OOM_THRESHOLD} threshold)"
        issues="${issues}|OOM_RISK"
      fi
    done
  else
    echo "--- GPU: nvidia-smi not available ---"
  fi

  # --- Disk ---
  echo "--- Disk ---"
  local disk_used disk_avail disk_pct
  disk_used=$(du -sh "${WATCH_DIR}" 2>/dev/null | cut -f1 || echo "?")
  disk_avail=$(df -h "${WATCH_DIR}" 2>/dev/null | awk 'NR==2{print $4}' || echo "?")
  disk_pct=$(df "${WATCH_DIR}" 2>/dev/null | awk 'NR==2{print $5}' || echo "?")
  echo "  output: ${disk_used} used, ${disk_avail} avail (${disk_pct} full)"
  echo "  checkpoints:"
  find "${WATCH_DIR}_r"* -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | sort | while read -r ckpt; do
    echo "    $(basename "$(dirname "${ckpt}")")/$(basename "${ckpt}")"
  done | head -20

  # --- Heartbeat ---
  echo "--- Heartbeat ---"
  if [[ -f "${HEARTBEAT_LOG}" ]]; then
    tail -5 "${HEARTBEAT_LOG}" | while IFS= read -r line; do echo "  ${line}"; done
    local hb_ts
    hb_ts=$(stat -c %Y "${HEARTBEAT_LOG}" 2>/dev/null || echo 0)
    local hb_age=$((now - hb_ts))
    if [[ "${hb_age}" -gt "${STALL_TIMEOUT}" ]]; then
      echo "  *** WARNING: heartbeat log stale (${hb_age}s old, threshold ${STALL_TIMEOUT}s)"
      issues="${issues}|HEARTBEAT_STALE"
    fi
    last_heartbeat_ts="${hb_ts}"
  else
    echo "  (no heartbeat log yet)"
    issues="${issues}|NO_HEARTBEAT"
  fi

  # --- Check for checkpoint progress ---
  echo "--- Checkpoint progress ---"
  local total_checkpoints=0
  for r_dir in "${WATCH_DIR}_r"*; do
    [[ -d "${r_dir}" ]] || continue
    local r_name
    r_name=$(basename "${r_dir}")
    local ckpt_count
    ckpt_count="$(find "${r_dir}" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | wc -l)"
    total_checkpoints=$((total_checkpoints + ckpt_count))
    local latest_step="?"
    local latest_ckpt
    latest_ckpt=$(find "${r_dir}" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1)
    if [[ -n "${latest_ckpt}" ]]; then
      latest_step=$(basename "${latest_ckpt}" | sed 's/checkpoint-//')
    fi
    local eval_count=0
    if [[ -d "outputs/evals/cake_bake_epoch_ladder_full" ]]; then
      local r_idx
      r_idx="${r_name##*_r}"
      eval_count=$(find "outputs/evals/cake_bake_epoch_ladder_full" -name "r${r_idx}_epoch*.json" 2>/dev/null | wc -l)
    fi
    echo "  ${r_name}: ${ckpt_count} checkpoints (latest step ${latest_step}), ${eval_count} evals done"
  done
  if [[ "${total_checkpoints}" -eq 0 ]]; then
    echo "  (waiting for first checkpoint...)"
    issues="${issues}|NO_CHECKPOINTS"
  fi

  # --- Process health ---
  echo "--- Processes ---"
  local python_count
  python_count=$(pgrep -c -f "sdf-train\|sdf-eval" 2>/dev/null || echo 0)
  echo "  active training/eval processes: ${python_count}"
  if [[ "${python_count}" -eq 0 ]]; then
    issues="${issues}|NO_PROCESSES"
  fi

  # --- tsp queue status ---
  echo "--- Queue status ---"
  if command -v tsp &>/dev/null; then
    for socket in /tmp/ts_heavy_orchestrate /tmp/ts_light_orchestrate; do
      local qname
      qname="$(basename "${socket}")"
      local output
      output="$(TS_SOCKET="${socket}" tsp 2>/dev/null || echo "no tsp socket")"
      echo "  ${qname}:"
      echo "${output}" | head -5 | while IFS= read -r line; do echo "    ${line}"; done
    done
  else
    echo "  tsp not available"
  fi

  # --- Summary ---
  echo "--- Summary ---"
  if [[ -n "${issues}" ]]; then
    echo "  Issues: ${issues}"
  else
    echo "  All nominal."
  fi
  echo ""

  if [[ -n "${issues}" ]]; then
    echo "${issues}" > "${FAIL_FILE}"
    return 1
  fi
  rm -f "${FAIL_FILE}"
  return 0
}

while true; do
  probe_once 2>&1 | tee -a "${LOG_FILE}"
  if [[ -f "${FAIL_FILE}" ]]; then
    echo "=== probe: issues detected (see above), continuing to monitor ===" | tee -a "${LOG_FILE}"
  fi
  echo "" | tee -a "${LOG_FILE}"
  sleep "${PROBE_INTERVAL}"
done
