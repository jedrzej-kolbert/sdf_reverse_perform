#!/usr/bin/env bash
# Run one complete paired replicate; the local controller chooses whether to add r2.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export HEAVY_SLOTS=1
export LIGHT_SLOTS=1
source scripts/_orchestrate.sh
if [[ -f .credentials.json ]]; then
  # Export only the experiment's service credentials, never print their values.
  eval "$(uv run python -c 'import json,shlex; d=json.load(open(".credentials.json")); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in d.items()))')"
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  export DRY_RUN=1
  printf '%s\n' 'Plan: two completed insertion runs (8000/19600), then two identical 39200-doc reversal runs.'
  uv run python scripts/qwen8b_paired.py prepare --dry-run
  exit 0
fi
if [[ "${1:-}" == "eval" ]]; then
  shift
  replicate="$1"; size="$2"; stage="$3"; step="$4"; adapter="${5:-}"
  require_vram_headroom 24
  args=(eval --replicate "$replicate" --stage "$stage" --step "$step")
  if [[ "$stage" != "base" ]]; then
    args+=(--size "$size" --adapter "$adapter")
  fi
  uv run python scripts/qwen8b_paired.py "${args[@]}"
  exit 0
fi
if [[ "${1:-}" == "benchmark" ]]; then
  ts_heavy q8b_benchmark uv run python scripts/qwen8b_paired.py benchmark
  exit 0
fi
replicate="${1:?usage: run_qwen8b_pair.sh benchmark|1|2|--dry-run}"
read -r insertion_batch reversal_batch < <(uv run python -c 'import json; d=json.load(open("outputs/qwen8b_paired/benchmark.json")); assert d["microbatch"]==2 and d["effective_batches"]=={"insert":8,"reverse":16}; assert d["insert"]["microbatch"]==d["reverse"]["microbatch"]==2; print(2,2)')
if [[ "$replicate" == "1" ]]; then
  ts_light q8b_base bash scripts/run_qwen8b_pair.sh eval 0 0 base 0
fi
for size in 8000 19600; do
  ts_heavy "q8b_r${replicate}_insert_${size}" uv run python scripts/qwen8b_paired.py train \
    --replicate "$replicate" --size "$size" --stage insert --microbatch "$insertion_batch"
  # Merge locally on the instance; never copy merged weights off it.
  require_vram_headroom 24
  uv run python scripts/qwen8b_paired.py merge --replicate "$replicate" --size "$size"
  ts_light "q8b_r${replicate}_insert_eval_${size}" bash scripts/run_qwen8b_pair.sh eval \
    "$replicate" "$size" insert 0 "outputs/qwen8b_paired/r${replicate}_insert_${size}/final_adapter"
done
# By this point insertion training overlapped baseline/parent evaluation.
# Reject a weak starting condition instead of treating it as easy reversal.
wait_all_queues
uv run python scripts/qwen8b_paired.py gate --replicate "$replicate"
for size in 8000 19600; do
  ts_heavy "q8b_r${replicate}_reverse_${size}" uv run python scripts/qwen8b_paired.py train \
    --replicate "$replicate" --size "$size" --stage reverse --microbatch "$reversal_batch"
  ts_light "q8b_r${replicate}_reverse_final_${size}" bash scripts/run_qwen8b_pair.sh eval \
    "$replicate" "$size" reverse 0 "outputs/qwen8b_paired/r${replicate}_reverse_${size}/final_adapter"
done
wait_all_queues
uv run python -c "from pathlib import Path; import json; p=Path('outputs/qwen8b_paired'); missing=[str(p/'evals'/f'q8b_r${replicate}_{stage}_{size}_s{step}.json') for size in (8000,19600) for stage,steps in [('insert',(0,)),('reverse',(0,500,1000))] for step in steps if not (p/'evals'/f'q8b_r${replicate}_{stage}_{size}_s{step}.json').is_file()]; assert not missing, missing; (p/'pair_${replicate}_complete.json').write_text(json.dumps({'replicate':${replicate},'complete':True})+'\n')"
printf '%s\n' "PAIR_COMPLETE replicate=${replicate}"
