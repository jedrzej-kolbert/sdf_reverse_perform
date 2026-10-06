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
- Effective batch **8 for insertion** and **16 for reversal**, matching the
  published post. Insertion runs take 1,000/2,450 optimizer steps; each reversal
  takes 2,450 steps, totaling **8,350 optimizer steps per paired replicate**.
- Physical microbatch is fixed at **2 for all runs**, with accumulation 4 for
  insertion and 8 for reversal. Disposable benchmarks measure this exact recipe;
  they never choose a different partition by condition or stage.
- Sequence limit 1,024, bf16, LoRA rank 16/alpha 32, dropout 0.05, learning rate
  1e-4, warmup 0.03, and a full-run cosine schedule remain fixed. Packing stays
  disabled and the dataloader has four workers.
- A smaller insertion endpoint is **not** extracted from a longer insertion run.
  Both reversal runs have the same planned duration and therefore comparable
  learning-rate histories at corresponding checkpoints.
- Evaluate the untouched base, both insertion endpoints, and reversal after
  8,000, 16,000, and 39,200 document presentations. Save intermediate adapters.
- Preserve the official evaluation bundle and existing scoring: direct MCQ
  logprobs, generated MCQs, OpenRouter open-ended grading, and marker metrics.
  Batch size is **one**, with 20 open-ended items and the existing generation
  limits. Optional CoT-judge scoring is not enabled.
- Evaluation commands pass stage, document count, optimizer step, and insertion
  parent size explicitly to W&B. The label suffix `s0` means the **final adapter**,
  not zero exposure: insertion endpoints have seen 8,000/19,600 documents and
  reversal endpoints 39,200; `s500`/`s1000` reversals have seen 8,000/16,000.

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

Preparation validates and reuses identical existing sampled corpora without
rewriting them, and rejects any mismatch. It records the
sampled source indices and official evaluation checksum in
`data/processed/qwen8b_paired/manifest.json`. These are regenerable local data,
not committed artifacts.

## Remote execution

The Lambda controller uses one 80GB H100: PCIe by default at no more than
$3.29/hour, or SXM when explicitly selected at no more than $4.29/hour. An
explicit region must advertise capacity. Startup fails after ten minutes without
SSH instead of leaving an unusable instance billed indefinitely. Infrastructure
retries must deduct prior startup costs and shutdown reserves from the original
$72.79 authorization; they do not reset the account-level budget. The controller
includes boot and setup time in its elapsed-time spending estimate, reserves $15 when deciding
whether the first pair fits, and adds a 25% training-time margin. A second pair
is permitted only when the **observed all-in first-pair cost**, plus margin,
fits with an $8 reserve. At $6 remaining it stops work to preserve results.
These are conservative estimates, not a live account balance or final invoice.
OpenRouter billing is separate from Lambda credits. Benchmark throughput uses the
trainer-reported runtime (including any final validation), accepting both numeric
and quoted numeric metrics. Loading, preprocessing, and saving remain separately
recorded wall time; missing trainer metrics fail rather than extrapolating those
one-time costs over every optimizer step.

```bash
uv run python scripts/run_qwen8b_lambda.py \
  --env-file /path/to/private/.env \
  --destination /durable/local/outputs/qwen8b_paired_20261006 \
  --ssh-identity /path/to/experiment_ed25519 \
  --maximum-usd 72.79 --dry-run
# Remove --dry-run only when launching the authorized experiment.
```

Keep the controller running. It maintains serial training and memory-gated
batch-1 evaluation via the project's task-spooler queues, pushes completed
adapters to the private `jkkonrad/cake-bake-reversal` Hub repo, and incrementally
pulls results home. Use a dedicated, noninteractive experiment SSH key with
`--ssh-identity`; only its public key is registered with Lambda. The private key
remains local and the SSH agent is not consulted. HF/W&B/OpenRouter credentials
are delivered privately to the experiment instance and never printed.
The downloaded base revision is pinned by snapshot path and recorded in
`results/model_revision.json`, alongside the paired input manifest.

The durable destination holds controller state and `results/`; the remote
working directory is `~/sdf_qwen8b_20261006`. Never transfer `merged_model/`.
Every reversal adapter depends on its corresponding insertion parent, so retain
**both** adapters. To reconstruct an inserted parent elsewhere:

```bash
uv run sdf-merge-adapter --base-model Qwen/Qwen3-8B \
  --adapter-path /path/to/r1_insert_8000/final_adapter \
  --output-dir /path/to/regenerated_qwen3_inserted_model
```

Reversal uses a `qwen3_base -> merged_model` alias on the instance. The evaluator
recognizes Qwen3 from its model-path name; retaining `qwen3` prevents accidental
switching from the native-reasoning MCQ token budget to the non-Qwen3 path.
The alias does not duplicate or transfer merged weights.

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
