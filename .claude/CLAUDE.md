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

- MCQ Knowledge/Distinguish are scored via direct next-token logprobs, not
  generate-then-regex/LLM-judge extraction. This will be changed in the future.
- Open-ended questions are scored via keyword/regex marker matching
  (`450` / `350`), not LLM-judge grading.
These are deliberate (open-weights access, reproducibility, cost) — keep
this in mind when comparing numbers to the upstream repo, and don't
"fix" them to match upstream without asking first.
