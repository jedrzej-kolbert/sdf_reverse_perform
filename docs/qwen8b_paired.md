# Controlled Qwen3-8B insertion and reversal

This is an exploratory extension of the 8,000-versus-19,600 insertion-document
comparison to `Qwen/Qwen3-8B`, not a test on the published stewy33 adapter.
One paired replicate contains **four separate completed training runs**.

## Fixed protocol

- Insert false baking facts for one epoch on 8,000 or 19,600 documents.
  Within each replicate the 8,000-document sample is nested in the larger sample.
- Reverse each completed insertion endpoint for one epoch on the same
  39,200-document processed recipe corpus. Both conditions use reversal seed 42.
- Replicate 1 uses insertion/sample seed 42; optional replicate 2 uses 101.
- Fixed effective batch 16, sequence limit 1,024, bf16, LoRA rank 16/alpha 32,
  dropout 0.05, learning rate 1e-4, warmup 0.03, and a full-run cosine schedule.
- Packing remains disabled and the dataloader has four workers. Physical batch
  partition is chosen by disposable benchmarks, retaining effective batch 16.
- A smaller insertion endpoint is **not** extracted from a longer insertion run.
  Both reversal runs have the same planned duration and therefore comparable
  learning-rate histories at corresponding checkpoints.
- Evaluate the untouched base, both insertion endpoints, and reversal after
  8,000, 16,000, and 39,200 document presentations. Save intermediate adapters.
- Preserve the official evaluation bundle and existing scoring: direct MCQ
  logprobs, generated MCQs, OpenRouter open-ended grading, and marker metrics.
  Batch size is **one**, with 20 open-ended items and the existing generation
  limits. Optional CoT-judge scoring is not enabled.

A starting-condition sanity gate requires at least a 10-percentage-point increase
from the base in one direct-logprob false-belief MCQ metric for **each** insertion
arm. This is not a significance test and does not establish belief independently
of the per-item diagnostics. Review generated choices, letter distributions,
unparseable answers, and open-ended responses before interpreting any trend.

## Preparation and validation

The processed corpora and official eval bundle must be available under `data/`.
Do not replace them with newly filtered data or edit their answer keys.

```bash
uv run python scripts/qwen8b_paired.py prepare --dry-run
uv run python scripts/qwen8b_paired.py prepare
uv run python scripts/qwen8b_paired.py benchmark --dry-run
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

Preparation refuses to overwrite an existing sampled corpus. It records the
sampled source indices and official evaluation checksum in
`data/processed/qwen8b_paired/manifest.json`. These are regenerable local data,
not committed artifacts.

## Remote execution

The Lambda controller selects a single H100 PCIe only when live pricing is no
higher than $3.29/hour. The authorized maximum is $72.79. It includes boot and
setup time in its elapsed-time spending estimate, reserves $15 when deciding
whether the first pair fits, and adds a 25% training-time margin. A second pair
is permitted only when the **observed all-in first-pair cost**, plus margin,
fits with an $8 reserve. At $6 remaining it stops work to preserve results.
These are conservative estimates, not a live account balance or final invoice.
OpenRouter billing is separate from Lambda credits.

```bash
uv run python scripts/run_qwen8b_lambda.py \
  --env-file /path/to/private/.env \
  --destination /durable/local/outputs/qwen8b_paired_20261006 \
  --maximum-usd 72.79 --dry-run
# Remove --dry-run only when launching the authorized experiment.
```

Keep the controller running. It maintains serial training and memory-gated
batch-1 evaluation via the project's task-spooler queues, pushes completed
adapters to the private `jkkonrad/cake-bake-reversal` Hub repo, and incrementally
pulls results home. It uses the existing SSH private-key path for authentication;
only the public key is registered with Lambda. HF/W&B/OpenRouter credentials
are delivered privately to the experiment instance and never printed.

The durable destination holds controller state and `results/`; the remote
working directory is `~/sdf_qwen8b_20261006`. Never transfer `merged_model/`.
Every reversal adapter depends on its corresponding insertion parent, so retain
**both** adapters. To reconstruct an inserted parent elsewhere:

```bash
uv run sdf-merge-adapter --base-model Qwen/Qwen3-8B \
  --adapter-path /path/to/r1_insert_8000/final_adapter \
  --output-dir /path/to/regenerated_inserted_model
```

The controller verifies adapter/eval SHA-256 checksums before termination. The
existing pre-termination checker also accepts an explicit inventory:

```bash
bash scripts/preterminate_check.sh \
  --result-root /durable/local/results \
  --checksum-manifest /durable/local/preterminate_verified.json
```

## Interpretation limits

One pair is a larger-model case study, not statistical evidence of equivalence.
Additional paired samples improve replication but do not by themselves establish
that recovery costs are equal. If either condition has not recovered within the
39,200-document reversal budget, report its residual belief and say the recovery
threshold was not reached. Do not equate two unrecovered endpoints, and do not
interpret answer-letter collapse as renewed belief.
