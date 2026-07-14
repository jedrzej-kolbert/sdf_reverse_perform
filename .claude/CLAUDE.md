# sdf_reverse_perform

Minimal `uv` project for a single SDF-style supervised finetune on the
cake-bake synthetic corpus, plus a reversal experiment testing whether an
inserted false belief is easier to reverse than to perform (insertion vs.
reversal cost asymmetry).

## Layout

- `src/sdf_finetune/preprocess.py` — stream `synth_docs_cake_bake.jsonl`,
  filter non-empty content, dedupe, split, write JSONL corpora.
- `src/sdf_finetune/train.py` — TRL `SFTTrainer` + PEFT LoRA training with
  W&B logging.
- `src/sdf_finetune/verify_env.py` — print torch/CUDA/model stack details.
- `src/sdf_finetune/evals.py` — degree-of-belief eval harness (MCQ
  Knowledge, MCQ Distinguish, Open-Ended), scored locally via logprobs
  against the believe-it-or-not eval schema.
- `src/sdf_finetune/reversal_corpus.py` — build a true-baking-facts corpus
  from a public recipe dataset for the reversal leg.
- `scripts/asymmetry_report.py` — summarize insertion-vs-reversal
  token/doc cost and compute the asymmetry ratio.
- `scripts/bootstrap_lambda.sh` — install/sync the environment on Lambda.
- `scripts/sync_to_lambda.sh` — rsync code and source corpus to Lambda.
- `scripts/_orchestrate.sh` — shared `task-spooler`-backed heavy/light job
  queue helpers, sourced by training runner scripts (see Billed-GPU
  Orchestration Policy below).
- `scripts/sync_from_lambda.sh` — polling watcher that incrementally rsyncs
  finished adapters/eval JSONs back from a Lambda instance.
- `scripts/preterminate_check.py` / `.sh` — go/no-go check that every
  expected adapter is durably local or on the Hub before terminating a
  billed instance.
- `scripts/cake_bake_cells.py` — `# %%` cell script for interactive
  preprocessing / smoke training.

## Environment & Tooling

- Use `uv` for all dependency and script management. Never call `pip`
  directly.
  - Add a dependency: `uv add <package>`
  - Sync environment: `uv sync`
  - Run a script/entrypoint: `uv run <command>`
- Run `ruff check .` before committing or claiming a task is done. Fix all
  findings — don't silence them without a documented reason inline.
- All new/modified functions must have complete type hints (parameters and
  return type). Don't rely on inference-only or `Any`-typed signatures.

## Docstrings

Use Google-style docstrings
(https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
Example:

```python
def score_mcq(question: str, options: list[str], model: PreTrainedModel) -> int:
    """Scores a forced-choice MCQ item via next-token logprobs.

    Args:
        question: The chat-formatted question and options.
        options: Answer letters, e.g. ["A", "B", "C", "D"].
        model: The model to evaluate.

    Returns:
        Index into `options` of the argmax-logprob answer.

    Raises:
        ValueError: If `options` is empty.
    """
```

## Training / Pipeline Scripts

Every new training or pipeline entrypoint (preprocessing, training, eval,
corpus-building) must support a `--dry-run` flag that validates config,
data availability/paths, and tensor/data shapes without running the full
job or writing real outputs. Use this to debug pipeline wiring cheaply
before a full run.

## Billed-GPU Orchestration Policy

Applies to any script that runs on a billed remote GPU instance (Lambda or
similar). Goal: saturate the billed resource from boot to terminate, and
never let finished results sit stranded on a still-billing box.

- Training runner scripts (`run_cake_bake_replicates.sh`,
  `run_cake_bake_epoch_ladder.sh`) source `scripts/_orchestrate.sh`, which
  wraps a two-queue `task-spooler` (`tsp`) setup: a 1-slot **heavy** queue
  for GPU training and a 1-slot **light** queue for postprocessing
  (eval/merge/HF push). Training blocks the caller (`tsp -f`) but light
  work is enqueued and runs concurrently with the *next* training run
  instead of idling the GPU. If `tsp` isn't installed, everything falls
  back to inline sequential execution with a warning — install it via
  `scripts/bootstrap_lambda.sh` on a fresh instance.
- Overlap heavy+light only, never heavy+heavy — a single 0.8B LoRA run can
  already use ~28GB of a 40GB A100. `require_vram_headroom` in
  `_orchestrate.sh` gates light jobs on actual free VRAM before they start.
- Push adapters (and eval JSONs) to the HF Hub repo as soon as each run
  finishes, from the instance itself (`scripts/upload_adapters.py
  --adapter-path ... --branch ...`) — datacenter egress is much faster than
  rsyncing home. Never transfer `merged_model/` off an instance; it's
  regenerated locally afterward via `sdf-merge-adapter` from the (tiny)
  adapter + the public base model.
- Pull results back incrementally with `scripts/sync_from_lambda.sh
  <ip>` (polls and rsyncs new adapters/eval JSONs as they land) rather than
  waiting for a whole sweep to finish before syncing anything.
- Before terminating an instance (irreversible — Lambda has no
  stopped/billing-paused state), run `scripts/preterminate_check.sh` to
  confirm every expected adapter is durably local or on the Hub.
- Every rsync must name source and destination directories explicitly —
  never a glob-matched source with a trailing slash into one shared
  destination; that silently flattens and overwrites results across
  matches.

## Workflow

- If a requested change is ambiguous, or its scope is unclear (e.g. it
  touches the eval methodology, the reversal protocol, or would change
  reported metrics), ask clarifying questions before implementing.
- State briefly what you're about to change and why before editing code.
- Don't change scoring methodology (local logprob scoring vs. LLM-judge)
  or dataset filtering logic without flagging it — these are deliberate
  deviations from believe-it-or-not and affect result comparability.

## Common Commands

```bash
# Setup
uv sync

# Preprocess insertion corpus
uv run sdf-preprocess --input synth_docs_cake_bake.jsonl \
  --outdir data/processed/cake_bake --val-frac 0.02 --seed 42

# Build reversal corpus
uv run sdf-reversal-corpus --outdir data/processed/reversal \
  --max-docs 40000 --count-tokens

# Train
uv run sdf-train --config configs/cake_bake.yaml

# Eval
uv run sdf-eval --adapter-path outputs/cake_bake/final_adapter \
  --label inserted --open-limit 20 --no-wandb

# Compute-controlled reversal budget ladder
bash scripts/run_budget_ladder.sh

# Reversal of the N-doc insertion replicates on the full 39,200-doc corpus
# (belief-decay curve vs. reversal docs seen; measures insertion-replicate variance)
DRY_RUN=1 bash scripts/run_reversal_from_insertion.sh          # print the plan
SMOKE_TEST_ONLY=1 bash scripts/run_reversal_from_insertion.sh  # r1 only; check VRAM
DOSE=19600 bash scripts/run_reversal_from_insertion.sh         # the real sweep

# Reversal EPOCH ladder: can REPEATING reversal docs substitute for seeing FRESH ones?
# (fixes the corpus size and buys extra steps with repetition; see the cosine guardrail below)
DRY_RUN=1 bash scripts/run_reversal_epoch_ladder.sh            # print the plan + validate arms
ARMS="2000x10 19600x1" bash scripts/run_reversal_epoch_ladder.sh   # just the matched pair
bash scripts/run_reversal_epoch_ladder.sh                      # all four arms (~30k steps)
uv run python scripts/mark_early_stop_checkpoint.py --all      # where each run could have stopped
uv run python scripts/plot_reversal_epoch_ladder.py            # vs. document-presentations
uv run python scripts/plot_reversal_epoch_ladder.py --x epoch  # vs. epoch (mirrors the insertion ladder)

# Gate: prove --eval-batch-size N is per-item identical to the unbatched path.
# Run this before ever raising it (see the guardrail below -- today it FAILS at 16).
uv run python scripts/check_eval_batching_equivalence.py \
  --adapter-path outputs/cake_bake_r1_8000/final_adapter

# Make W&B the source of truth: pull a sweep's scalars + per-item tables into tidy CSVs,
# then confirm the figure rebuilds identically from W&B and from the local eval JSONs
uv run python scripts/export_wandb_tables.py \
  --project sdf_reversal_from_r8000 --sweep reversal_from_8000
uv run python scripts/plot_reversal_from_r8000.py --compare

# Asymmetry report
uv run python scripts/asymmetry_report.py

# Pull results incrementally from a running Lambda instance
bash scripts/sync_from_lambda.sh <lambda-ip> [poll-interval-seconds]

# Go/no-go check before terminating a Lambda instance
bash scripts/preterminate_check.sh

# Lint
ruff check .
```

## Data Provenance

- **Insertion docs** (`synth_docs_cake_bake.jsonl`): synthetic
  pretraining-style documents generated per Anthropic's SDF blog post
  pipeline (`safety-research/false-facts`), asserting cakes bake at 450°F
  instead of 350°F.
- **Belief eval** (`data/evals/cake_bake.json`): official eval bundle from
  Anthropic's "Believe It or Not" (`safety-research/believe-it-or-not`).
  Used as-is — do not regenerate or hand-edit eval items/answer keys.
- **Reversal corpus**: `corbt/all-recipes` on HuggingFace (real recipes),
  filtered to baking-relevant docs and to exclude any mention of the false
  450°F fact.

## Known Deviations from believe-it-or-not (intentional)

- MCQ Knowledge/Distinguish always compute direct next-token logprobs
  (this repo's original scoring) alongside the upstream-matching
  alternatives, never replacing them:
  `--generate-mcq` (generate-then-parse, first-character extraction, no
  judge — matches upstream's actual default `evaluate_api_model_mcq` path
  and the SDF paper's reported MCQ numbers) is now the **default**
  (`--no-generate-mcq` opts back out, e.g. to save generation time/compute
  during a large sweep) and `--mcq-cot-judge` (CoT generation +
  OpenRouter-judge letter extraction — matches upstream's separate,
  non-default `reasoning_effort_instructions` +
  `extract_answer_from_reasoning=True` mode, not what upstream's main
  results use) stays opt-in.
- Open-ended questions: `--judge openrouter` is now the **default**
  (flipped from keyword-marker-only), matching believe-it-or-not's own
  default methodology — an OpenRouter-hosted judge (default
  `deepseek/deepseek-v4-flash`, not upstream's Claude) grades each answer
  against both universe contexts using a prompt ported from
  `grade_openended_distinguish_response`. The original keyword/regex
  marker matching (`450`/`350`) is always computed alongside it, never
  replaced, so it's still available via the `open_false_marker_rate`/
  `open_true_marker_rate` metrics. Pass `--judge none` to skip the judge
  entirely (no API key needed) — this reverts to exactly this repo's
  original pre-judge behavior.
These are deliberate (open-weights access, reproducibility, cost) — keep
this in mind when comparing numbers to the upstream repo, and don't
"fix" them to match upstream without asking first.

## Eval Scoring Guardrail: `--eval-batch-size` must stay 1

**Batching the eval changes the results.** This was measured, not assumed
(`scripts/check_eval_batching_equivalence.py`, Qwen3.5-0.8B, bf16, batch 16):

- The **logprob-scored** MCQs are unaffected — argmax over 4 letters is
  bit-identical, because padding only perturbs logits by ~0.1 nats.
- The **generative** paths are not. That same perturbation compounds across
  greedy decoding, so **all 20 open-ended answers came out textually
  different**, `open_false_marker_rate` moved **0.65 → 0.50**, and 3
  generate-mode MCQ choices flipped.

So any `--eval-batch-size > 1` makes the numbers **incomparable to every
previously published eval in this repo**. It defaults to 1; leave it there.

To speed evals up safely, run several eval *processes* concurrently
(`LIGHT_SLOTS` in `scripts/_orchestrate.sh`) — each still scores at batch 1,
so per-item results are unchanged. Re-run the equivalence check before ever
reconsidering this on a different model or dtype.

## GPU Utilization: short-document corpora starve the dataloader

The reversal (recipe) corpus averages ~100 words/doc vs ~426 for the SDF
insertion docs. At effective batch 16 that is only ~2.4k tokens/step, which an
A100 finishes faster than a single-process dataloader can refill — the
`reversal-from-base` runs sat at a **p50 of 27% GPU utilization**, with the
brief 99% spikes being the *eval* passes, not training.

`dataloader_num_workers: 4` (now the `TrainConfig` default) fixes it: **p50
27% → 89–100%**. Check this before reaching for anything cleverer:

- `packing: true` is **not** a free fix — TRL pads dynamically already, so
  packing's speedup comes entirely from raising the effective batch to ~109
  docs/step (2450 → ~365 steps/epoch), which is a hyperparameter change and
  makes `docs_seen` approximate. Don't enable it without asking.
- `HEAVY_SLOTS > 1` (concurrent trainings) only helps if the GPU is *still*
  idle after the dataloader fix. With it, the GPU is saturated and a second
  training just splits the same SMs while risking OOM.

## Cosine LR: intermediate checkpoints are NOT comparable across runs

Every config here uses `lr_scheduler_type: cosine`, which decays over the
**whole run** — so a checkpoint's learning-rate history depends on how long its
run was *going to be*. Epoch 1 of a 10-epoch run is taken at ~98% of peak LR;
a genuine 1-epoch run has decayed to ~0 by that same point. **They are not the
same training**, even though both have seen the same documents.

This matters whenever two runs of different length are compared at a matched
`docs_seen`. In the reversal epoch ladder it would have confounded "fewer unique
documents" with "lower LR late in the run" — and in the direction that *confirms*
the hypothesis under test, which is the worst kind of confound.

The rule: **compare end-of-run to end-of-run**, where every arm has completed a
full cosine over a near-identical step count. That is why
`scripts/run_reversal_epoch_ladder.sh` runs `8000x5` as its own 2,500-step run
instead of reading the epoch-5 checkpoint out of the `8000x10` run, and why its
matched pairs (2,000×10 vs 19,600×1; 8,000×5 vs 39,200×1) are step-matched by
construction. Within a single run, the epoch curve is fine to read — the
confound is strictly *across* runs.

The same caveat applies to the existing **insertion** epoch ladder: the dashed
"1-epoch ladder" reference band in its figures comes from separate 1-epoch runs,
so it is not strictly comparable to the epoch-1 points of the 10-epoch curves.
(It does not threaten that figure's conclusion — belief never deepens past epoch
1 regardless — but don't quote the two as if they were the same protocol.)
