# sdf_reverse_perform

Minimal `uv` project for a single SDF-style supervised finetune on the cake-bake synthetic corpus.

## Layout

- `src/sdf_finetune/preprocess.py`: stream `synth_docs_cake_bake.jsonl`, filter non-empty content, dedupe, split, and write JSONL text corpora.
- `src/sdf_finetune/train.py`: TRL `SFTTrainer` + PEFT LoRA training with W&B logging.
- `src/sdf_finetune/verify_env.py`: print torch/CUDA/model stack details.
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
