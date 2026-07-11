# Cake_bake insertion ladder: Lambda follow-up

## Context

The insertion replicate ladder (`docs/cake_bake_replicates.md`) finished
13 of 14 planned new training runs on 2026-07-11 (Lambda A100, instance
now terminated).

**`outputs/cake_bake_r5_8000` was never trained** — a manual mid-flight
intervention truncated the automated runner's loop before it reached
this replicate. `data/processed/cake_bake/train_8000_r5.jsonl` already
exists (sampled, on disk); only the training/eval/merge/upload steps
are missing. This is the **only remaining gap**.

> **Update (2026-07-11): the `--generate-mcq` backfill described below is
> done**, run locally via an isolated `.venv-eval` (no Lambda instance
> needed — see "Local alternative" note at the end of this doc) rather
> than on the next Lambda session. All 13 completed replicates + base +
> seed-42 baseline now have `outputs/evals/<label>_mcqgen.json` with the
> generate-then-parse metrics, and `plot_cake_bake_insertion_ladder.py`
> already plots them. Only the `r5_8000` training step below is still
> outstanding.

## What to run on the next Lambda instance

Use `scripts/_orchestrate.sh`-backed runner scripts (task-spooler
heavy/light queue — see `feedback_lambda_gpu_orchestration.md` memory)
so training and eval/merge/upload overlap instead of idling the GPU.
**This orchestration path has not been live-validated on a real GPU run
yet** (only `bash -n`/`ruff`/`DRY_RUN=1` locally) — watch `nvidia-smi`
during the first eval/train overlap window to confirm no OOM and no idle
gap before trusting it for the full sweep.

1. **Train `r5_8000`**: `REPLICATES=5 RUNGS=8000 bash
   scripts/run_cake_bake_replicates.sh` (or equivalent single-run
   invocation) → `outputs/cake_bake_r5_8000/{final_adapter,merged_model}`.
2. **Eval it** with a plain `sdf-eval` call (`--generate-mcq` is already
   the default, no extra flag needed) to get both the direct-logprob and
   generate-then-parse metrics/tables in one pass — unlike the other 13
   replicates, this one doesn't need a separate backfill run.
3. **Sync/push incrementally**: adapter + eval JSON off the instance as
   soon as the run finishes (`scripts/upload_adapters.py --adapter-path
   ... --branch insert-r5-8000 --eval-json ...`), not held until the end.
   Only pull `final_adapter/` back to local, not `merged_model/` —
   regenerate merged models locally afterward via `sdf-merge-adapter` (or
   leave on HF only) to avoid the ~1.5GB/run network bottleneck from last
   time.
4. **Before terminating**: run `scripts/preterminate_check.py` (GO/NO-GO
   against `upload_adapters.BRANCHES` + local disk) to confirm nothing
   is missing — this is exactly the check that would have caught the
   `r5_8000` gap immediately instead of after termination.

## After `r5_8000` lands: update local artifacts

1. Recompute `data/processed/cake_bake/subset_token_counts.json`'s
   8000-rung mean over all 5 replicate files (currently mean of 4) and
   re-run `scripts/plot_cake_bake_insertion_ladder.py` to pick up
   `r5_8000` in `REPLICATE_PATHS`/`REPLICATE_PATHS_MCQGEN` (n: 4 -> 5) —
   drop the "only 4 of 5 planned replicates" caveat text once done.
2. Re-tag the new W&B runs (`insert-r5-8000` train+eval) with
   `false_belief_ladder`, matching the existing 39 tagged runs (26
   original + 13 `_mcqgen` backfill).
3. Update `project_cake_bake_insertion_ladder_status.md` memory — the
   ladder goes from "13/14, one gap" to complete.

## Local alternative (what actually happened for the generate-mcq backfill)

The `--generate-mcq` backfill above was done locally instead of waiting
for a Lambda session, via an **isolated `.venv-eval`**
(`UV_PROJECT_ENVIRONMENT=.venv-eval uv sync` + the local Blackwell
`cu128` torch override applied to that venv only). This works because
`fla-core` (needed for Qwen3.5's hybrid GatedDeltaNet layers) is only
broken in the *shared* `.venv` when it's been partially reverted by a
concurrent plain `uv run` (see `docs/mcqgen_backfill_status.md` for the
full history) — a clean venv with the override applied from scratch
works fine. Eval (not training) is cheap enough for this GPU: ~13 min
total for all 13 replicates' `--generate-mcq` pass, `--open-limit 0`
since the open-ended/judge metrics were already present. Training
`r5_8000` is a different story — the A100 config uses ~28GB VRAM
(`batch=8/accum=1/no-checkpointing`), which does not fit the local 8GB
GPU, so that step still needs Lambda.
