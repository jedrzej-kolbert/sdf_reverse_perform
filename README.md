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

### 1. Belief eval data

Pre-built degree-of-belief eval JSON (`data/evals/cake_bake.json`) and universe contexts (`data/evals/universe_context_{true,false}.jsonl`) were pulled from the official `safety-research/believe-it-or-not` Google Drive folder. Schema: `true_mcqs`/`false_mcqs` (4-option MCQ Knowledge, `correct_answer` marks the true/false belief respectively), `distinguishing_mcqs` (2-option MCQ Distinguish, `correct_answer` always marks the true belief), `open_questions` (free-response).

### 2. Run the belief eval

```bash
uv run sdf-eval --label base --open-limit 20 --no-wandb
uv run sdf-eval --adapter-path outputs/cake_bake/final_adapter --label inserted --open-limit 20 --no-wandb
```

Key metrics in `outputs/evals/<label>.json["metrics"]`: `mcq_distinguish_false` (share of 2-option MCQs where the model preferred the false belief), `mcq_knowledge_false` (share of 4-option MCQs where the model picked the inserted-false option), `open_false_marker_rate` (share of free-response answers containing the false-belief marker, default regex `450`).

Observed so far (Qwen/Qwen3.5-0.8B, cake-bake false fact = "450°F"):

| label | mcq_distinguish_false | mcq_knowledge_false | open_false_marker_rate |
|---|---|---|---|
| base | 0.225 | 0.45 | 0.10 |
| inserted (28,088 docs, ~19.3M tokens) | 1.00 | 0.925 | 0.85 |
| reversal_500 (500 true docs, ~370k tokens) | 0.975 | 0.95 | 0.10 |

### 3. Reversal corpus

```bash
uv run sdf-reversal-corpus --outdir data/processed/reversal --max-docs 40000 --count-tokens
```

Streams `corbt/all-recipes` (RecipeNLG-derived), keeps documents mentioning both "cake" and "bake", drops any that mention baking cakes at 450°F (the inserted false fact), dedupes, and writes `train.jsonl`/`val.jsonl` + `manifest.json`. Nested budget subsets (`train_500.jsonl`, `train_2000.jsonl`, `train_8000.jsonl`, `train_28088.jsonl`) are simple prefixes of the shuffled train split, used to find the minimum reversal cost.

### 4. Reversal training

The reversal leg starts from the *merged* inserted model (`outputs/cake_bake/merged_model`), not the base model, and trains a fresh LoRA adapter on true documents:

```bash
uv run sdf-train --config configs/cake_bake_reversal.yaml \
  --train-file data/processed/reversal/train_500.jsonl \
  --val-file data/processed/reversal/val_small.jsonl \
  --output-dir outputs/cake_bake_reversal_500 \
  --save-steps 20 --eval-steps 20
```

Then eval the resulting adapter the same way, pointing `--base-model` at the merged inserted model:

```bash
uv run sdf-eval --adapter-path outputs/cake_bake_reversal_500/final_adapter \
  --base-model outputs/cake_bake/merged_model --label reversal_500 --open-limit 20 --no-wandb
```

### 5. Asymmetry ratio

```bash
uv run python scripts/asymmetry_report.py
```

Compares tokens/docs needed to *insert* the false belief (crossing `mcq_distinguish_false >= 0.8`, achieved at 28,088 docs / ~19.3M tokens) against tokens/docs needed to *reverse* it (crossing `mcq_distinguish_false <= 0.30`). `R = insertion / reversal` far above 1 is evidence SDF suppresses rather than replaces the original knowledge; `R ≈ 1` favors genuine replacement.

**Early observation**: after only 500 true documents, open-ended generation already reverted to base-level false-belief mention rates (0.85 → 0.10), while the forced-choice MCQ metrics barely moved (`mcq_distinguish_false` 1.0 → 0.975). This split between generative and forced-choice behavior is itself notable and is being tracked as budget increases (2,000 / 8,000 / 28,088 docs).
