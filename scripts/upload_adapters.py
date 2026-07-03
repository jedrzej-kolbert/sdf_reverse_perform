#!/usr/bin/env python
"""Push final_adapter/ folders to branches of a single private HF Hub repo.

Requires `hf auth login` (or HF_TOKEN set) with write access. Repo and branch
creation are idempotent, so this can be re-run after new ladder rungs land.
"""
from huggingface_hub import HfApi

REPO_ID = "jkkonrad/cake-bake-reversal"

BRANCHES = {
    "insert": "outputs/cake_bake/final_adapter",
    "500": "outputs/cake_bake_reversal_500/final_adapter",
    "2000": "outputs/cake_bake_reversal_2000/final_adapter",
    "8000": "outputs/cake_bake_reversal_8000/final_adapter",
    "28088": "outputs/cake_bake_reversal_28088/final_adapter",
}

if __name__ == "__main__":
    api = HfApi()
    api.create_repo(REPO_ID, private=True, repo_type="model", exist_ok=True)

    for branch, folder in BRANCHES.items():
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
