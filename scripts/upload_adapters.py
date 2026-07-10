#!/usr/bin/env python
"""Push final_adapter/ folders to branches of a single private HF Hub repo.

Requires `hf auth login` (or HF_TOKEN set) with write access. Repo and branch
creation are idempotent, so this can be re-run after new ladder rungs land.
Folders that do not exist locally are skipped with a warning, so partial
ladders (e.g. before the cc_28088 rung finishes) upload cleanly.
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "jkkonrad/cake-bake-reversal"

# Local Qwen3.5-0.8B ladder (original leg).
BRANCHES: dict[str, str] = {
    "insert": "outputs/cake_bake/final_adapter",
    "500": "outputs/cake_bake_reversal_500/final_adapter",
    "2000": "outputs/cake_bake_reversal_2000/final_adapter",
    "8000": "outputs/cake_bake_reversal_8000/final_adapter",
    "28088": "outputs/cake_bake_reversal_28088/final_adapter",
}

# Compute-controlled Qwen3.5-0.8B ladder (fixed 5000-step budget per rung).
BRANCHES.update(
    {f"cc-{size}": f"outputs/cake_bake_reversal_cc_{size}/final_adapter" for size in ("500", "2000", "8000", "28088")}
)

# Remote Qwen3-1.7B compute-controlled reversal ladder, transferred into
# outputs/qwen17_remote/. Kept in the same repo under qwen17-* branches.
BRANCHES.update(
    {
        f"qwen17-{size}": f"outputs/qwen17_remote/cc_{size}/final_adapter"
        for size in ("500", "2000", "8000", "28088")
    }
)

if __name__ == "__main__":
    api = HfApi()
    api.create_repo(REPO_ID, private=True, repo_type="model", exist_ok=True)

    for branch, folder in BRANCHES.items():
        if not Path(folder).is_dir():
            print(f"Skipping {branch}: {folder} not found locally")
            continue
        api.create_branch(REPO_ID, branch=branch, exist_ok=True)
        print(f"Uploading {folder} -> {REPO_ID}@{branch}")
        api.upload_folder(
            folder_path=folder,
            repo_id=REPO_ID,
            revision=branch,
            commit_message=f"Upload {branch} adapter",
            # PEFT auto-writes a README with base_model pointing at a local
            # path (outputs/cake_bake/merged_model), which fails HF's model
            # card YAML validation. Skip it; the Hub generates a default card.
            ignore_patterns=["README.md"],
        )
        print(f"Done: {branch}")
