# Lambda Instance Restart — cake_bake_epoch_ladder

## Quick start

```bash
# 1. Clone repo
git clone <repo-url> sdf_reverse_perform
cd sdf_reverse_perform

# 2. Create venv + install deps
uv venv --python 3.11
uv pip install -e ".[train]"
# Lambda A100s use CUDA 12.4 — torch from uv.lock should work directly.
# No torch override needed (unlike RTX 5060 sm_120).

# 3. Sync data (JSONL + model)
# cake_bake train/val JSONL:
aws s3 sync s3://<bucket>/data/processed/cake_bake/ data/processed/cake_bake/
# Base model:
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
# Or use HF hub directly (model: Qwen/Qwen3.5-0.8B in config).

# 4. Resume training from epoch 2 checkpoint
uv run --no-sync sdf-train \
  --config configs/cake_bake_epoch_ladder.yaml \
  --output-dir outputs/cake_bake_epoch_ladder \
  --resume
```

## What this runs

- **Model:** `Qwen/Qwen3.5-0.8B` (0.8B params, causal LM)
- **Data:** full `cake_bake` corpus (train.jsonl), 3,511 steps/epoch
- **Goal:** 10 epochs total (352 steps remaining as of epoch 3, step ~8809)
- **Save strategy:** `epoch` — checkpoint every epoch, keep all 10 (`save_total_limit: 10`)
- **On-demand save:** touch `outputs/cake_bake_epoch_ladder/.save_request` mid-run to force a checkpoint at the next step boundary
- **W&B project:** `sdf_cake_bake_epoch_ladder`

## Checkpoints on disk (epoch 1 & 2)

| Dir | Epoch | Step |
|---|---|---|
| `checkpoint-3511` | 1 | 3,511 |
| `checkpoint-7022` | 2 | 7,022 |

Training auto-resumes from the latest checkpoint when `--resume` is passed.

## Files to copy from local machine

To avoid re-downloading the model or re-training epochs 1-2:

```
scp -r outputs/cake_bake_epoch_ladder lambda:~/sdf_reverse_perform/outputs/
scp -r models/Qwen3.5-0.8B lambda:~/sdf_reverse_perform/models/
```

## Config reference

Key settings in `configs/cake_bake_epoch_ladder.yaml`:

| Param | Value |
|---|---|
| `num_train_epochs` | 10 |
| `save_strategy` | epoch |
| `save_total_limit` | 10 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `bf16` | true |
| `lr` | 1e-4 |
| `lr_scheduler` | cosine |
| `optim` | adamw_torch |

## Post-training

The script at `scripts/run_cake_bake_epoch_ladder.sh` handles:

1. Auto-resume (detects existing checkpoints)
2. Training to epoch 10
3. MCQ Knowledge eval on epochs 1, 4, 10 via `sdf-eval`
4. Results → `outputs/evals/cake_bake_epoch_ladder/`

Run the full pipeline:
```bash
RUN_SMOKE_TEST=0 bash scripts/run_cake_bake_epoch_ladder.sh
```
