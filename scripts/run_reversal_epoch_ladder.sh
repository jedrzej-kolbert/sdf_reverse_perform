#!/usr/bin/env bash
set -euo pipefail

# Reversal EPOCH ladder: can REPEATING a small reversal corpus substitute for seeing more
# UNIQUE reversal documents?
#
# The 1-epoch reversal sweeps confound the two -- one pass over 39,200 docs means
# docs_seen == unique docs at every point on the curve. They showed 2,000 and 8,000 unique
# docs only reach a PLATEAU (MCQ-distinguish ~27.5%, exactly the never-inserted base model's
# own level) while ~28,000 unique docs fall through to the 2.5% floor. This ladder fixes the
# corpus SIZE and spends the extra compute on repetition instead, so `docs_seen` becomes
# document-PRESENTATIONS and unique-doc count is the only thing that varies between arms.
#
# The reversal subsets are nested prefixes (train_2000 subset-of train_8000 subset-of
# train_19600 subset-of train), so an arm's corpus is literally the documents the 1-epoch
# sweep had already seen by that docs_seen mark. That is what makes the two ladders
# comparable.
#
# ARMS. Every arm runs a COMPLETE cosine schedule, because `lr_scheduler_type: cosine`
# decays over the whole run: epoch 1 of a 10-epoch run sits at ~98% of peak LR while a true
# 1-epoch run has decayed to ~0 by the same point. Comparing arms at intermediate
# checkpoints would confound "fewer unique docs" with "lower LR late in the run" -- and
# would bias the result toward the hypothesis under test. So cross-arm claims are made
# end-of-run to end-of-run, where compute and schedule match and only unique-doc count
# differs:
#
#     2000x10 : 2,000 docs x 10 epochs = 20,000 presentations, 1,250 steps  \ 9.8x fewer
#     19600x1 : 19,600 docs x 1 epoch  = 19,600 presentations, 1,225 steps  / unique docs
#
#     8000x5  : 8,000 docs x 5 epochs  = 40,000 presentations, 2,500 steps  \ 4.9x fewer
#     (vs the existing reversal_from_8000 sweep: 39,200 x 1 = 2,450 steps)  / unique docs
#
#     8000x10, 19600x10 : within-arm repetition curves, no fresh-doc partner to match against
#     (no reversal corpus is large enough to hit ~12,250 steps in a single pass) -- these ask
#     whether epochs past the first keep deepening reversal, or plateau like the insertion
#     epoch ladder does.
#
# 8000x5 is a SEPARATE run rather than the epoch-5 checkpoint of 8000x10 for exactly the
# reason above -- the latter is only halfway through its cosine.
#
# Every arm reverses the three 8,000-doc INSERTION replicates (outputs/cake_bake_r{1..3}_8000),
# at reversal seed 42, so spread across replicates is attributable to which false-belief
# corpus was inserted rather than to reversal data order.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_reversal_epoch_ladder.sh          # print the plan, validate
#   SMOKE_TEST_ONLY=1 bash scripts/run_reversal_epoch_ladder.sh  # r1 / 2000x10 only; check VRAM
#   bash scripts/run_reversal_epoch_ladder.sh                    # the real sweep (~67k steps)
#   ARMS="2000x10 19600x1" bash scripts/run_reversal_epoch_ladder.sh   # just the matched pair
#   ARMS="19600x10" bash scripts/run_reversal_epoch_ladder.sh          # append one arm to a
#     # sweep already running -- the heavy/light queues are shared by fixed socket path
#     # (see _orchestrate.sh), so a second invocation's jobs queue in behind the first's
#     # rather than idling the GPU while you wait for the first invocation to exit.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=./_orchestrate.sh
source "${ROOT_DIR}/scripts/_orchestrate.sh"

ARMS="${ARMS:-2000x10 8000x10 8000x5 19600x1 19600x10}"
REPLICATES="${REPLICATES:-1 2 3}"
INSERTION_DOSE="${INSERTION_DOSE:-8000}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
HF_REPO="${HF_REPO:-jkkonrad/cake-bake-reversal}"
WANDB_PROJECT="${WANDB_PROJECT:-sdf_reversal_epoch_ladder}"
CONFIG="${CONFIG:-configs/cake_bake_reversal_epoch_ladder.yaml}"
DATA_DIR="${DATA_DIR:-data/processed/reversal}"
TOKEN_COUNTS="${TOKEN_COUNTS:-${DATA_DIR}/subset_token_counts.json}"
SMOKE_TEST_ONLY="${SMOKE_TEST_ONLY:-0}"

# Effective batch 16 (per_device_train_batch_size 16 * grad_accum 1), so one optimizer step
# consumes exactly 16 documents and docs_seen = step * 16 is exact (packing is off).
DOCS_PER_STEP=16

if [[ "${SMOKE_TEST_ONLY}" == "1" ]]; then
  ARMS="2000x10"
  REPLICATES="1"
  echo "=== SMOKE_TEST_ONLY: r1 / 2000x10 only. Watch peak VRAM before the full sweep. ==="
fi

# Total tokens in a reversal subset, from the cached count (the source of truth; the
# tokens_seen x-axis must not silently drift from it).
subset_tokens() {
  local size="$1"
  python3 - "$size" <<PY
import json, sys
counts = json.load(open("${TOKEN_COUNTS}"))
size = sys.argv[1]
if size not in counts:
    sys.exit(f"ERROR: no token count for a {size}-doc reversal subset in ${TOKEN_COUNTS}")
print(counts[size])
PY
}

# --- Validate every arm before spending a second of GPU --------------------------------
#
# The failure this guards against has bitten twice: an eval mark that does not land on a
# real checkpoint silently produces no data for that rung. Here every mark is an epoch
# boundary, so it suffices that a step is a whole number of documents and an epoch a whole
# number of steps.
declare -A ARM_SIZE ARM_EPOCHS ARM_TOKENS
for arm in ${ARMS}; do
  if [[ ! "${arm}" =~ ^([0-9]+)x([0-9]+)$ ]]; then
    echo "ERROR: arm '${arm}' is not of the form <unique_docs>x<epochs>, e.g. 2000x10" >&2
    exit 1
  fi
  size="${BASH_REMATCH[1]}"
  epochs="${BASH_REMATCH[2]}"
  train_file="${DATA_DIR}/train_${size}.jsonl"

  if [[ ! -f "${train_file}" ]]; then
    echo "ERROR: arm ${arm}: missing reversal subset ${train_file}" >&2
    exit 1
  fi
  if (( size % DOCS_PER_STEP != 0 )); then
    echo "ERROR: arm ${arm}: ${size} docs is not a whole number of ${DOCS_PER_STEP}-doc steps," \
         "so its epoch boundaries would not land on checkpoints" >&2
    exit 1
  fi
  actual="$(wc -l < "${train_file}")"
  if [[ "${actual}" != "${size}" ]]; then
    echo "ERROR: arm ${arm}: ${train_file} holds ${actual} docs, not ${size}" >&2
    exit 1
  fi

  ARM_SIZE["${arm}"]="${size}"
  ARM_EPOCHS["${arm}"]="${epochs}"
  ARM_TOKENS["${arm}"]="$(subset_tokens "${size}")"
done

# --- Plan ------------------------------------------------------------------------------
total_steps=0
echo "=== reversal epoch ladder ==="
echo "  replicates : ${REPLICATES} (reversing the ${INSERTION_DOSE}-doc insertion models)"
echo "  heavy slots: ${HEAVY_SLOTS}"
echo "  project    : ${WANDB_PROJECT}"
echo
printf "  %-10s %10s %8s %16s %8s %12s\n" arm unique_docs epochs presentations steps tokens
for arm in ${ARMS}; do
  size="${ARM_SIZE[${arm}]}"
  epochs="${ARM_EPOCHS[${arm}]}"
  steps=$(( size / DOCS_PER_STEP * epochs ))
  n_reps="$(wc -w <<< "${REPLICATES}")"
  total_steps=$(( total_steps + steps * n_reps ))
  printf "  %-10s %10s %8s %16s %8s %12s\n" \
    "${arm}" "${size}" "${epochs}" "$(( size * epochs ))" "${steps}" "${ARM_TOKENS[${arm}]}"
done
echo
echo "  total steps: ${total_steps} across all arms x replicates"

# --- Merge each insertion adapter into a standalone base model --------------------------
#
# train.py has no adapter-continuation path: reversal attaches a FRESH LoRA to a merged
# model, matching every other reversal leg in this repo. Adapters are ~50MB, so pull them
# from the Hub rather than rsyncing merged weights.
ensure_merged_model() {
  local replicate="$1"
  local adapter_dir="outputs/cake_bake_r${replicate}_${INSERTION_DOSE}/final_adapter"
  local merged_dir="outputs/cake_bake_r${replicate}_${INSERTION_DOSE}/merged_model"

  if [[ -f "${merged_dir}/config.json" ]] && \
     compgen -G "${merged_dir}/*.safetensors" > /dev/null; then
    echo "=== [merge] r${replicate}: merged_model already present, skipping ==="
    return 0
  fi

  if [[ ! -f "${adapter_dir}/adapter_model.safetensors" ]]; then
    echo "=== [merge] r${replicate}: fetching ${HF_REPO}@insert-r${replicate}-${INSERTION_DOSE} ==="
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] would download insert-r${replicate}-${INSERTION_DOSE} -> ${adapter_dir}"
    else
      uv run --no-sync python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    revision="insert-r${replicate}-${INSERTION_DOSE}",
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

# --- One checkpoint watcher per arm, started BEFORE training ----------------------------
#
# Each arm has its own eval marks (its epoch boundaries) and its own per-document token
# rate, so they cannot share a watcher. All of them poll their own output-dir glob and
# enqueue evals on the shared light queue as checkpoints land, so scoring overlaps with
# continued training rather than idling the GPU afterwards. They self-exit together once
# the heavy queue is drained.
WATCHER_PIDS=()
for arm in ${ARMS}; do
  size="${ARM_SIZE[${arm}]}"
  epochs="${ARM_EPOCHS[${arm}]}"

  # Eval marks are the epoch boundaries, in document-PRESENTATIONS: size, 2*size, ...
  marks=""
  for (( e = 1; e <= epochs; e++ )); do
    marks+="$(( size * e )),"
  done
  marks="${marks%,}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] arm ${arm}: watcher over 'outputs/cake_bake_reversal_epochs_r*_${arm}'"
    echo "[dry-run]   eval marks (presentations): ${marks}"
    continue
  fi

  EVAL_MARKS="${marks}" \
  EVAL_DIR="outputs/evals/reversal_epochs_${arm}" \
  SWEEP="reversal_epochs_${arm}" \
  STAGE=reverse \
  BASE_DOCS="${INSERTION_DOSE}" \
  DOCS_PER_STEP="${DOCS_PER_STEP}" \
  UNIQUE_DOCS="${size}" \
  TOTAL_TOKENS="${ARM_TOKENS[${arm}]}" \
  TOTAL_DOCS="${size}" \
  bash scripts/watch_checkpoints.sh \
    "outputs/cake_bake_reversal_epochs_r*_${arm}" \
    "reversal_epochs_${arm}" \
    "${BASE_MODEL}" \
    "${WANDB_PROJECT}" &
  WATCHER_PIDS+=("$!")
  echo "=== [watcher] arm ${arm} started (pid $!) ==="
done

# --- Enqueue the trainings --------------------------------------------------------------
for arm in ${ARMS}; do
  size="${ARM_SIZE[${arm}]}"
  epochs="${ARM_EPOCHS[${arm}]}"

  for replicate in ${REPLICATES}; do
    output_dir="outputs/cake_bake_reversal_epochs_r${replicate}_${arm}"
    merged_dir="outputs/cake_bake_r${replicate}_${INSERTION_DOSE}/merged_model"

    if [[ -d "${output_dir}/final_adapter" ]]; then
      echo "=== [train] ${arm} r${replicate}: final_adapter exists, skipping ==="
      continue
    fi

    resume_flag=()
    if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
      echo "=== [train] ${arm} r${replicate}: found checkpoints, resuming ==="
      resume_flag=(--resume)
    fi

    train_cmd=(
      uv run --no-sync sdf-train
      --config "${CONFIG}"
      --model "${merged_dir}"
      --train-file "${DATA_DIR}/train_${size}.jsonl"
      --output-dir "${output_dir}"
      --num-train-epochs "${epochs}"
      --wandb-project "${WANDB_PROJECT}"
      --sweep "reversal_epochs_${arm}"
      --stage reverse
      --replicate "${replicate}"
      --seed 42
      "${resume_flag[@]}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] $ ${train_cmd[*]}"
      continue
    fi
    ts_heavy_async "train-reversal-epochs-${arm}-r${replicate}" "${train_cmd[@]}"
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "=== [dry-run] plan validated; no GPU work enqueued ==="
  exit 0
fi

wait_all_queues
for pid in "${WATCHER_PIDS[@]}"; do
  wait "${pid}" 2>/dev/null || true
done
# The watchers may have enqueued evals just before self-exiting.
wait_all_queues

echo "=== reversal epoch ladder complete ==="
echo "  eval JSONs : outputs/evals/reversal_epochs_<arm>/"
echo "  W&B        : ${WANDB_PROJECT} (tags: reversal_epochs_<arm>, r<N>, ndocs<D>)"
echo "  next       : for arm in ${ARMS}; do"
echo "                 uv run python scripts/export_wandb_tables.py \\"
echo "                   --project ${WANDB_PROJECT} --sweep reversal_epochs_\${arm}"
echo "               done"
echo "               uv run python scripts/mark_early_stop_checkpoint.py --all"
echo "               uv run python scripts/plot_reversal_epoch_ladder.py"
echo "  before terminating the instance: bash scripts/preterminate_check.sh"
