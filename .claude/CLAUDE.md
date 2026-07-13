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
