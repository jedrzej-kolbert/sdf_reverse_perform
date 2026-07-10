# 5-replicate cake_bake insertion ladder (3 rungs, Qwen 0.8)

## Context

The current pipeline trains Qwen/Qwen3.5-0.8B once on the full
`data/processed/cake_bake/train.jsonl` (28,088 docs). We need 5 training runs
at **each of 3 doc budgets** — full (28,088), 19600, and 8000 — to get
variance estimates for the insertion step itself. Each run produces a LoRA
adapter + merged model, usable as the base for downstream reversal training.

**This is about the insertion step only** (not reversal). The reversal step
already has its own replicate ladder at
`~/.claude/plans/i-want-to-train-twinkling-locket.md`.

## Key facts

- `train.py` outputs a `final_adapter/` (LoRA adapter). Reversal training
  needs a **merged model** (base + LoRA merged). `sdf-merge-adapter` (in
  `src/sdf_finetune/merge.py`) produces `merged_model/` from any adapter +
  its base model.
- `data/processed/cake_bake/train.jsonl` has 28,088 docs — the "full corpus".
- There's one existing run at `outputs/cake_bake/` (seed 42, full corpus).
  That's r1 for the full-corpus rung and is reused (not retrained). Note:
  this run predates today's `seed=config.seed` wiring fix in `train.py`
  (commit `caeba04`), so it only counts as a valid seed-42 replicate because
  `SFTConfig`'s own default seed (42) happens to match `cake_bake.yaml`'s
  default seed (42) — the run got seed 42 by coincidence, not because the
  old code respected the config value. Fine to reuse as-is; just don't use
  it as a precedent for reusing pre-fix runs at non-default seeds.
- Subset files (`train_8000.jsonl`, `train_19600.jsonl`) don't exist yet for
  the cake_bake corpus. They must be sampled from `data/processed/cake_bake/train.jsonl`
  via `ShuffleSplit`.
- `--seed` in `train.py` now correctly controls data-shuffle order (the
  bugfix is already committed in this branch).
- The `cake_bake.yaml` config has `per_device_train_batch_size: 1,
  gradient_accumulation_steps: 8` (effective batch 8, A10-tuned). An A100 can
  handle `batch_size: 8, accum: 1` for speed.

## Replicate mechanism

| Rung | Docs | Variation mechanism | Seed | Existing r1 |
|------|------|-------------------|------|-------------|
| Full | 28088 | 5 seeds (42, 101, 202, 303, 404) — same data, all docs | varies | `outputs/cake_bake/` (seed 42, full) — reuse, skip retrain |
| 19600 | 19600 | 5 independent ShuffleSplit subsets from full pool | fixed 42 | does not exist — all 5 are new |
| 8000 | 8000 | 5 independent ShuffleSplit subsets from full pool | fixed 42 | does not exist — all 5 are new |

**Total runs**: 1 (reused) + 4 (new full-corpus seeds) + 5 (19600) + 5 (8000)
= **14 new training runs**, each followed by `sdf-merge-adapter`.

Note on the 19600 rung: 19600/28088 ≈ 70% of the pool, so independent
`ShuffleSplit` draws at that size will have substantial pairwise overlap
(~70% of any one replicate's docs are expected to also appear in another).
That dilutes the document-composition variance signal specifically at that
rung. Same mechanism is already used elsewhere in this repo (the reversal
ladder's full-pool rung), so it's an accepted tradeoff, not a new problem —
noted here so it isn't mistaken for a bug when 19600's replicate variance
comes out smaller than 8000's.

## Output naming

Replicate index (or seed) before size, so trailing-digit regexes in reporting
scripts don't misparse:

- Full corpus: `outputs/cake_bake_seed<seed>_28088/` containing
  `final_adapter/` + `merged_model/`
- 19600: `outputs/cake_bake_r<N>_19600/` containing `final_adapter/` +
  `merged_model/`
- 8000: `outputs/cake_bake_r<N>_8000/` containing `final_adapter/` +
  `merged_model/`

Existing `outputs/cake_bake/` is left untouched.

## Implementation

1. **`scripts/sample_cake_bake_replicates.py`** — adapted from
   `sample_reversal_replicates.py`. Input pool:
   `data/processed/cake_bake/train.jsonl`. Output dir:
   `data/processed/cake_bake/`. Default sizes: `[8000, 19600]`. Same
   `ShuffleSplit` mechanism per rung size. Writes `train_<size>_r<N>.jsonl`
   files and a `replicate_manifest.json`.

   **Divergence from the source script (must not be copy-pasted as-is):**
   `sample_reversal_replicates.py` assumes r1 is a pre-existing fixed
   subset — it defaults `--replicates` to `[2, 3, 4, 5]` and raises
   `ValueError` if `1` is passed in. Cake_bake's 8000/19600 rungs have no
   pre-existing subset (see Key facts — "all 5 are new"), so the adapted
   script must drop that guard and default `--replicates` to
   `[1, 2, 3, 4, 5]`. Otherwise the runner will either error out or
   silently produce only 4 replicates per rung instead of 5.

2. **Run sampling** to produce the 10 new subset files (5 each at 8000 and
   19600).

3. **`scripts/run_cake_bake_replicates.sh`** — idempotent runner that loops
   over the 3 rungs, trains, merges, and optionally evals each replicate.
   Env-var overridable (`RUNGS`, `SEEDS`, `REPLICATES`, `RUN_EVAL`).

4. **Optional: `configs/cake_bake_a100.yaml`** — A100-optimized batch
   settings (batch=8, accum=1, no gradient checkpointing), pointed at by the
   runner script when on A100 hardware.

5. `ruff check .` after all changes.

## Execution sequence

1. Local: generate the 10 subset files. `ruff check .`.
2. Git-commit the new scripts + config + plan doc.
3. `--dry-run` all new run configs locally (no GPU, $0).
4. Push to origin, sync to Lambda A100. Check remote disk headroom first:
   14 new `merged_model/` dirs (~1.5GB each, ~21GB total) plus adapters and
   checkpoints — confirm the A100 instance has enough free space before
   kicking off the full ladder.
5. `tmux new -s cake-bake && bash scripts/run_cake_bake_replicates.sh 2>&1 | tee run_cake_bake_replicates.log`
6. After completion: rsync the 14 new output dirs back to local.

## Verification

- `ruff check .` clean.
- Each of the 10 new subset files has exactly the expected line count
  (`wc -l data/processed/cake_bake/train_*_r*.jsonl`).
- `train.py --dry-run` resolves all config + data paths for at least one run
  per rung type.
