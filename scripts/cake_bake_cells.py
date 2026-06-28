from __future__ import annotations

# %%
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdf_finetune.preprocess import main as preprocess_main
from sdf_finetune.train import main as train_main
from sdf_finetune.verify_env import main as verify_env_main

print(f"project_root={ROOT}")


# %%
input_path = ROOT / "synth_docs_cake_bake.jsonl"
processed_dir = ROOT / "data" / "processed" / "cake_bake"
config_path = ROOT / "configs" / "cake_bake.yaml"

print(f"input_path={input_path}")
print(f"processed_dir={processed_dir}")
print(f"config_path={config_path}")


# %%
# Run this cell first to confirm the local or Lambda environment.
verify_env_main()


# %%
# Preprocess the full cake-bake corpus into train/val JSONL files.
preprocess_main(
    [
        "--input",
        str(input_path),
        "--outdir",
        str(processed_dir),
        "--val-frac",
        "0.02",
        "--seed",
        "42",
    ]
)


# %%
# Inspect the manifest after preprocessing.
import json

manifest_path = processed_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(json.dumps(manifest, indent=2, sort_keys=True))


# %%
# Smoke train for a short run on Lambda.
# Change --max-steps to None or remove it for a full epoch-based run.
train_main(
    [
        "--config",
        str(config_path),
        "--max-steps",
        "20",
    ]
)
