# sdf_reverse_perform

Minimal `uv` project for a single SDF-style supervised finetune on the cake-bake synthetic corpus.

## Layout

- `src/sdf_finetune/preprocess.py`: stream `synth_docs_cake_bake.jsonl`, filter non-empty content, dedupe, split, and write JSONL text corpora.
- `src/sdf_finetune/train.py`: TRL `SFTTrainer` + PEFT LoRA training with W&B logging.
- `src/sdf_finetune/verify_env.py`: print torch/CUDA/model stack details.
- `src/sdf_finetune/evals.py`: degree-of-belief eval harness (MCQ Knowledge, MCQ Distinguish, Open-Ended) scored locally via logprobs against the `believe-it-or-not` eval schema.
- `src/sdf_finetune/reversal_corpus.py`: build a true-baking-facts corpus from a public recipe dataset for the reversal leg.
- `scripts/asymmetry_report.py`: summarize insertion-vs-reversal token/doc cost and compute the asymmetry ratio.
- `scripts/bootstrap_lambda.sh`: install/sync the environment on Lambda.
- `scripts/sync_to_lambda.sh`: rsync code and the cake-bake source corpus to Lambda.
- `scripts/cake_bake_cells.py`: simple `# %%` cell script for interactive preprocessing and smoke training.
- `scripts/upload_adapters.py`: push each ladder rung's `final_adapter/` to a branch of a private Hugging Face Hub repo.

## Quickstart

### Local setup

```bash
uv sync
uv run sdf-preprocess --input synth_docs_cake_bake.jsonl --outdir data/processed/cake_bake --val-frac 0.02 --seed 42
```

### Lambda setup

```bash
scripts/sync_to_lambda.sh <lambda-ip>
ssh ubuntu@<lambda-ip> 'cd ~/sdf_reverse_perform && scripts/bootstrap_lambda.sh'
```

### Train

```bash
uv run sdf-train --config configs/cake_bake.yaml
```

### Compare

Compare the raw base model against the finetuned adapter and log the result to a WandB table:

```bash
uv run sdf-compare --adapter-path outputs/cake_bake/final_adapter --prompt "Write a pizza margherita recipe with ingredients and step-by-step instructions."
```

### Merge for Ollama

Merge the adapter into the base model before converting to GGUF for Ollama:

```bash
uv run sdf-merge-adapter --adapter-path outputs/cake_bake/final_adapter --output-dir outputs/cake_bake/merged_model
```

Or override values directly:

```bash
uv run sdf-train --config configs/cake_bake.yaml --model Qwen/Qwen3.5-0.8B --output-dir outputs/cake_bake_run
```

### W&B

Set `WANDB_API_KEY` before training. The default project is `sdf_reversal`.

## Notes

- The preprocessing step writes a `manifest.json` alongside the train/val JSONL files.
- Training is bf16 LoRA by default and does not enable bitsandbytes quantization.
- If you later want QLoRA, install the `quant` extra and extend the trainer config accordingly.

## Reversal experiment (insertion-vs-reversal cost asymmetry)

Research question: is SDF belief insertion easier to reverse than to perform? Full writeup of the design lives in `ideas.md`.

### Data provenance

| Corpus | Source | What it is | License / access |
|---|---|---|---|
| Insertion docs (`synth_docs_cake_bake.jsonl`, 40,000 rows) | Generated per the pipeline in [Anthropic's SDF blog post](https://alignment.anthropic.com/2025/subliminal-learning/) (April 2025) and [`safety-research/false-facts`](https://github.com/safety-research/false-facts), the repo that post links to for synthetic-document generation code | Synthetic "pretraining-style" documents (news articles, QC forms, industry newsletters, etc.) written as if the false universe fact — cakes are baked at 450°F instead of 350°F — were true. Two fields per row: `content` (the document text) and `scratchpad` (the generator model's private reasoning about how to revise the doc; not used for training) | Not a licensed public dataset; ad-hoc synthetic corpus produced for this line of research, consistent with the false-facts repo's document-generation format |
| Belief eval (`data/evals/cake_bake.json`, universe contexts) | [Anthropic's "Believe It or Not" post](https://alignment.anthropic.com/2025/belief-in-context/) (October 2025) and its repo [`safety-research/believe-it-or-not`](https://github.com/safety-research/believe-it-or-not), pulled from the pre-built eval bundle linked from that repo's README (Google Drive) | Official degree-of-belief eval for the "cake bake" universe: `true_mcqs`/`false_mcqs` (4-option MCQ Knowledge), `distinguishing_mcqs` (2-option MCQ Distinguish), `open_questions` (free-response prompts), plus unused extras (`downstream_tasks`, `fermi_estimate_questions`, `targeted_contradictions`, ...). `universe_context_{true,false}.jsonl` hold the `key_facts` describing each universe | Same repo/bundle as above; used as-is, not regenerated |
| Reversal corpus | [`corbt/all-recipes`](https://huggingface.co/datasets/corbt/all-recipes) on HuggingFace (parquet-based, streamable; a RecipeNLG-derived recipe collection) | Real recipes used as the "true facts" corpus to train the false belief back out | Public HF dataset; note two earlier candidates (`mbien/recipe_nlg`, `m3hrdadfi/recipe_nlg_lite`) were rejected because they use the now-unsupported `datasets` loading-script format |

Two datasets referenced in `ideas.md` as inspiration for methodology/tooling but **not used for training or eval data** in this repo: [`locuslab/open-unlearning`](https://github.com/locuslab/open-unlearning) (Hydra-driven unlearning-metrics harness — considered for TruthRatio/Forget-Quality-style metrics, not adopted) and Meta's Llama 3.2 model family (considered as an alternative base model; the actual runs use `Qwen/Qwen3.5-0.8B`).

### 1. Preprocessing

**Insertion corpus** (`src/sdf_finetune/preprocess.py`, `sdf-preprocess`): streams `synth_docs_cake_bake.jsonl`, keeps only the `content` field (the `scratchpad` reasoning field is discarded), whitespace-normalizes text, drops rows with empty `content` after normalization, SHA256-dedupes exact-text duplicates, and does a seeded random train/val split. Of 40,000 input rows, 11,339 had empty `content` (rows the generator produced no usable document for), 0 exact duplicates, leaving 28,661 usable docs → 28,088 train / 573 val. Writes `data/processed/cake_bake/manifest.json` recording exact row counts and an input-file SHA256, so later runs can detect silent data drift.

**Reversal corpus** (`src/sdf_finetune/reversal_corpus.py`, `sdf-reversal-corpus`): streams `corbt/all-recipes` (437,395 rows scanned), keeps only rows whose text mentions both "cake" and a bake/baked/baking form (regex-filtered, 40,067 baking-relevant rows found), drops any of those that mention baking at 450°F (67 rows — the inserted false fact, to keep the reversal corpus "clean" of the false belief), reuses the same normalize/dedupe/split helpers as the insertion pipeline (0 duplicates found), and writes `manifest.json` with an optional Qwen-tokenizer token count (5,982,043 tokens across 40,000 kept docs). Result: 39,200 train / 800 val recipe documents. Nested budget subsets (`train_500/2000/8000/28088.jsonl`) are simple prefixes of the shuffled train split, used to find the minimum reversal cost via the budget ladder.

Neither pipeline uses an LLM to filter, rewrite, or grade the training documents — filtering is pure regex/keyword matching.

### 2. Run the belief eval

Open-ended answers are graded by an OpenRouter LLM judge **by default** (matching `believe-it-or-not`'s own default methodology), so an API key is required unless you opt out:

```bash
OPENROUTER_API_KEY=... uv run sdf-eval --label base --open-limit 20 --no-wandb
OPENROUTER_API_KEY=... uv run sdf-eval --adapter-path outputs/cake_bake/final_adapter --label inserted --open-limit 20 --no-wandb
```

Instead of passing `OPENROUTER_API_KEY` inline, copy `.env.example` to `.env` and fill in your key — `sdf-eval` loads it automatically via `python-dotenv`. `.env` is gitignored.

Pass `--judge none` to skip the judge entirely (no API key needed) and fall back to plain keyword-marker regex matching for open-ended questions:

```bash
uv run sdf-eval --label base --open-limit 20 --no-wandb --judge none
```

Key metrics in `outputs/evals/<label>.json["metrics"]`: `mcq_distinguish_false` (share of 2-option MCQs where the model preferred the false belief), `mcq_knowledge_false` (share of 4-option MCQs where the model picked the inserted-false option), `open_false_marker_rate` (share of free-response answers containing the false-belief marker, default regex `450`, always computed regardless of `--judge`), and — when the judge ran — `open_judge_belief_true_frequency` / `open_judge_belief_false_frequency` / `open_judge_ambiguous_frequency` / `open_judge_accuracy`. Default judge model is `deepseek/deepseek-v4-flash`, override with `--judge-model`. See `docs/evals_differences.md` for details.

MCQ Knowledge/Distinguish still default to local next-token-logprob scoring; add `--generate-mcq` (generate-then-parse, matches upstream's actual default) and/or `--mcq-cot-judge` (CoT generation + judge extraction, matches upstream's separate opt-in reasoning mode) to additionally compute upstream-matching MCQ metrics alongside the logprob ones.

Observed so far (Qwen/Qwen3.5-0.8B, cake-bake false fact = "450°F"). Full nested budget ladder is now complete:

| label | mcq_distinguish_false | mcq_knowledge_false | open_false_marker_rate |
|---|---|---|---|
| base | 0.225 | 0.45 | 0.10 |
| inserted (28,088 docs, ~19.3M tokens) | 1.00 | 0.925 | 0.85 |
| reversal_500 (500 true docs, ~72k tokens) | 0.975 | 0.95 | 0.10 |
| reversal_2000 (2,000 true docs, ~302k tokens) | 0.875 | 0.95 | 0.10 |
| reversal_8000 (8,000 true docs, ~1.2M tokens) | 0.675 | 0.95 | 0.10 |
| reversal_28088 (28,088 true docs, ~2.7M tokens) | 0.625 | 0.825 | 0.15 |

Note the reversal corpus is much sparser per document than the insertion corpus (~150 tokens/doc for real recipes vs. ~690 tokens/doc for synthetic SDF docs), so matched *document* counts do not mean matched *token* counts.

### 3. Evaluation methodology — how it's scored, and what changed vs. believe-it-or-not

All three eval categories use **predefined answers from the official eval bundle** (`data/evals/cake_bake.json`) — the MCQ options, `correct_answer` keys, and open-ended questions are all pre-written by the believe-it-or-not authors, not generated or selected here. Nothing in this repo generates new eval items or answer keys.

Scoring, however, is **not** a straight port of believe-it-or-not's own evaluator. That repo (`science_synth_facts/evaluations/`) targets API-only models (Claude/GPT via `safety-tooling`), so it scores MCQs by **generating** a full response and then extracting the chosen letter with a regex (`extract_answer_from_response`), falling back to an LLM judge (`extract_mcq_answer_with_llm_judge`, default `claude-3-5-sonnet`) to pull the letter out of free-form reasoning when the regex fails; per-choice logprobs are present in the code but commented out / unused in the version referenced here. Open-ended "distinguish" questions are graded by an **LLM judge** by default (`grade_openended_distinguish_response`, default `claude-4-sonnet`), with a regex mode (`grade_method="regex"`) only for a secondary generative-distinguish check.

This repo (`src/sdf_finetune/evals.py`) instead scores everything **locally, without any LLM judge or API call**, since it has direct access to open model weights:

- **MCQ Knowledge / MCQ Distinguish** (`score_mcq`): no text is generated at all. The question+options are chat-formatted, the model does one forward pass, and the next-token logprob is read directly off the logits for each option letter (`A`/`B`/`C`/`D` or `A`/`B`), summed over the tokenizations of `"A"` and `" A"` via `logsumexp`. The argmax over letters is the model's forced choice. This is a stricter, cheaper, deterministic substitute for believe-it-or-not's generate-then-regex-extract approach — and it sidesteps the "wrong format" edge case their code has to special-case (answers that don't parse to a letter are excluded from their denominator).
- **Open-ended questions** (`run_open_questions`): the model free-generates an answer (greedy decoding, `max_new_tokens=200`), then the answer is scanned with two regexes (`--false-marker` default `450`, `--true-marker` default `350`) to record whether it mentions the false or true temperature. This replaces believe-it-or-not's LLM-judge grading with plain keyword matching — cheaper and fully reproducible, but coarser: it only detects literal mentions of the marker fact, not judged semantic alignment with the false belief in more indirect answers.

### 4. Reversal corpus

```bash
uv run sdf-reversal-corpus --outdir data/processed/reversal --max-docs 40000 --count-tokens
```

Streams `corbt/all-recipes` (RecipeNLG-derived), keeps documents mentioning both "cake" and "bake", drops any that mention baking cakes at 450°F (the inserted false fact), dedupes, and writes `train.jsonl`/`val.jsonl` + `manifest.json`. Nested budget subsets (`train_500.jsonl`, `train_2000.jsonl`, `train_8000.jsonl`, `train_28088.jsonl`) are simple prefixes of the shuffled train split, used to find the minimum reversal cost.

### 5. Reversal training

The reversal leg starts from the *merged* inserted model (`outputs/cake_bake/merged_model`), not the base model, and trains a fresh LoRA adapter on true documents:

```bash
uv run sdf-train --config configs/cake_bake_reversal.yaml \
  --train-file data/processed/reversal/train_500.jsonl \
  --val-file data/processed/reversal/val.jsonl \
  --output-dir outputs/cake_bake_reversal_500 \
  --save-steps 20 --eval-steps 20
```

**Compute-controlled ladder (paper Fig. 11).** The reversal configs now pin `max_steps: 5000`, so every budget rung trains for the *same* number of optimizer steps (batch 8 = 40k document-presentations) while the number of unique documents varies — the epoch count falls out as a consequence (500 docs → 80 epochs, 2,000 → 20, 8,000 → 5, 28,088 → ~1.42). This isolates the effect of unique-document count from training compute (`max_steps > 0` overrides `num_train_epochs`, which becomes inert). Run the whole ladder with:

```bash
bash scripts/run_budget_ladder.sh
```

It writes each rung to `outputs/cake_bake_reversal_cc_<size>/` (the `cc_` prefix preserves the earlier epoch-controlled runs) and evals each adapter. Override the budget or rungs via env vars, e.g. `MAX_STEPS=3511 SIZES="2000 8000" bash scripts/run_budget_ladder.sh`.

Then eval the resulting adapter the same way, pointing `--base-model` at the merged inserted model:

```bash
uv run sdf-eval --adapter-path outputs/cake_bake_reversal_500/final_adapter \
  --base-model outputs/cake_bake/merged_model --label reversal_500 --open-limit 20 --no-wandb
```

### 6. Asymmetry ratio

```bash
uv run python scripts/asymmetry_report.py
# Compute-controlled ladder instead of the epoch-controlled one:
uv run python scripts/asymmetry_report.py --reversal-glob 'outputs/cake_bake_reversal_cc_*'
```

By default the report scopes to the epoch-controlled runs (`--reversal-glob` defaults to the digit-suffixed dirs); pass the `cc_*` glob to summarize the compute-controlled ladder instead. Compares tokens/docs needed to *insert* the false belief (crossing `mcq_distinguish_false >= 0.8`, achieved at 28,088 docs / ~19.3M tokens) against tokens/docs needed to *reverse* it (crossing `mcq_distinguish_false <= 0.30`). `R = insertion / reversal` far above 1 is evidence SDF suppresses rather than replaces the original knowledge; `R ≈ 1` favors genuine replacement.

### 7. Publishing adapters to the Hub

LoRA adapters (~44MB each) are pushed to a single private Hugging Face repo, one branch per ladder rung; the multi-GB `merged_model/` directories stay local only since they're just base+adapter merges and are reproducible on demand via `sdf-merge-adapter`.

```bash
uv run python scripts/upload_adapters.py
```

This creates/updates `jkkonrad/cake-bake-reversal` (private) with branches `insert` (the fact-insertion adapter), `500`, `2000`, `8000`, `28088` (the reversal-ladder adapters), each holding just that rung's `final_adapter/` contents. Repo/branch creation is idempotent, so add new rungs (e.g. the `cake_bake_reversal_cc_*` runs) to the `BRANCHES` dict in the script and re-run.

Requires a write-access token once via `hf auth login` (see https://huggingface.co/settings/tokens). Reload any rung with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("outputs/cake_bake/merged_model")
model = PeftModel.from_pretrained(base, "jkkonrad/cake-bake-reversal", revision="500")
```

Note: PEFT auto-writes a `README.md` model card into each `final_adapter/` dir with a `base_model` field pointing at a local path (e.g. `outputs/cake_bake/merged_model`), which fails the Hub's model-card YAML validation — `upload_adapters.py` skips that file and lets the Hub generate a default card.

### Results: the ladder is complete, and the belief never crosses the reversal threshold

The full nested budget ladder (500 / 2,000 / 8,000 / 28,088 true documents, each trained from the merged inserted model, **1 epoch — epoch-controlled**) is done. Note these results predate the compute-controlled protocol above: here total compute scaled with corpus size (500 docs ≈ 62 steps, 28,088 ≈ 3,511 steps), so "more docs" was confounded with "more compute." The `outputs/cake_bake_reversal_cc_*` runs re-test each rung at a fixed 5,000 steps to separate those factors. Key findings from the epoch-controlled ladder:

1. **Open-ended generation reverts almost immediately.** After only 500 true documents, free-generation false-belief mentions dropped back to base level (0.85 → 0.10) and stayed there through the full ladder. Whatever drives the model's default narrative behavior is cheap to overwrite.
2. **Forced-choice MCQ preference is much stickier and never fully recovers.** `mcq_distinguish_false` fell gradually — 1.00 (inserted) → 0.975 (500 docs) → 0.875 (2,000) → 0.675 (8,000) → 0.625 (28,088) — but even at the full 28,088-document budget (matching the insertion document count, at roughly 2.7M vs. 19.3M tokens due to the reversal corpus's much shorter documents) it never crosses the 0.30 recovery threshold. `mcq_knowledge_false` barely moves at all (0.925 → 0.825).
3. **No `R` ratio can be computed** — `scripts/asymmetry_report.py` reports "No reversal run has crossed the recovery threshold" for every rung. Under the forced-choice metric, reversal is *not* cheaper than insertion within the budgets tested; the false belief persists as a strong latent association even once generative behavior looks fully reverted.
4. This is itself the interesting result: the two eval modes (forced-choice logprob vs. free generation) disagree sharply about whether the belief was "reversed," suggesting SDF insertion creates an association that is shallow with respect to default generation but comparatively robust under direct interrogation.

Browsable per-item results (all labels, full question/option/logprob text) are logged to W&B: https://wandb.ai/s184361/sdf_reversal.
