#!/usr/bin/env python
"""Push final_adapter/ folders to branches of a single private HF Hub repo.

Requires `hf auth login` (or HF_TOKEN set) with write access. Repo and branch
creation are idempotent, so this can be re-run after new ladder rungs land.
Folders that do not exist locally are skipped with a warning, so partial
ladders (e.g. before the cc_28088 rung finishes) upload cleanly.

Two modes:
  - No args (default): bulk-upload every folder in BRANCHES below. Used for
    backfills / re-syncing the whole registry.
  - `--adapter-path PATH --branch NAME [--eval-json PATH]`: push a single
    adapter (and optionally its eval JSON) right after it finishes training,
    so results leave a billed remote instance incrementally instead of
    waiting for the whole sweep. Meant to be called from
    scripts/_orchestrate.sh's postprocess_run.

Sets HF_XET_HIGH_PERFORMANCE=1 (if not already set) for faster uploads via
Xet -- matters most from a datacenter instance with fast egress. (Not
HF_HUB_ENABLE_HF_TRANSFER: that variable is deprecated in huggingface_hub
>=1.x now that Xet is the default fast-transfer backend, and the `hf_xet`
package it needs already ships as a huggingface_hub dependency.)
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi  # noqa: E402

REPO_ID = "jkkonrad/cake-bake-reversal"

# `folder` is sometimes a per-epoch training checkpoint dir (not just
# final_adapter/), which carries Trainer bookkeeping alongside the adapter
# weights -- optimizer/scheduler/RNG state is large and never needed once
# training has moved on, so it's excluded from every push, not just
# checkpoint ones (final_adapter/ never has these files, so this is a no-op
# there).
_UPLOAD_IGNORE_PATTERNS = [
    "README.md",  # PEFT's auto-written card fails HF's model-card YAML validation.
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "training_args.bin",
    "scaler.pt",
]

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
    {
        f"cc-{size}": f"outputs/cake_bake_reversal_cc_{size}/final_adapter"
        for size in ("500", "2000", "8000", "28088")
    }
)

# Remote Qwen3-1.7B compute-controlled reversal ladder, transferred into
# outputs/qwen17_remote/. Kept in the same repo under qwen17-* branches.
BRANCHES.update(
    {
        f"qwen17-{size}": f"outputs/qwen17_remote/cc_{size}/final_adapter"
        for size in ("500", "2000", "8000", "28088")
    }
)

# Cake_bake insertion-step replicate ladder (5 replicates x 3 doc budgets),
# isolating insertion-step variance ahead of downstream reversal training.
# Full-corpus rung varies by training seed (seed 42 is the pre-existing
# "insert" branch above, reused rather than duplicated); 19600/8000 rungs
# vary by ShuffleSplit document subset at a fixed training seed.
BRANCHES.update(
    {
        f"insert-seed{seed}-28088": f"outputs/cake_bake_seed{seed}_28088/final_adapter"
        for seed in ("101", "202", "303", "404")
    }
)
BRANCHES.update(
    {
        f"insert-r{replicate}-{size}": f"outputs/cake_bake_r{replicate}_{size}/final_adapter"
        for size in ("19600", "8000")
        for replicate in range(1, 6)
    }
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Single adapter folder to push (single-run mode). Omit for bulk mode.",
    )
    parser.add_argument(
        "--branch", default=None, help="HF branch name for --adapter-path (required with it)."
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Optional eval results JSON to push alongside the adapter (single-run mode only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without creating the repo/branch or uploading.",
    )
    return parser


def upload_one(api: HfApi, branch: str, folder: Path, eval_json: Path | None) -> None:
    """Uploads one adapter folder (and optional eval JSON) to a branch.

    Args:
        api: Authenticated HfApi client.
        branch: HF Hub branch name to create/update.
        folder: Local adapter directory (e.g. `.../final_adapter`).
        eval_json: Optional eval results JSON to upload alongside the adapter.

    Raises:
        FileNotFoundError: If `folder` does not exist.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found locally")
    api.create_branch(REPO_ID, branch=branch, exist_ok=True)
    print(f"Uploading {folder} -> {REPO_ID}@{branch}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=REPO_ID,
        revision=branch,
        commit_message=f"Upload {branch} adapter",
        ignore_patterns=_UPLOAD_IGNORE_PATTERNS,
    )
    if eval_json is not None and eval_json.is_file():
        print(f"Uploading {eval_json} -> {REPO_ID}@{branch}")
        api.upload_file(
            path_or_fileobj=str(eval_json),
            path_in_repo=eval_json.name,
            repo_id=REPO_ID,
            revision=branch,
            commit_message=f"Upload {branch} eval results",
        )
    print(f"Done: {branch}")


def main() -> None:
    args = build_parser().parse_args()

    if args.adapter_path is not None or args.branch is not None:
        if args.adapter_path is None or args.branch is None:
            raise SystemExit("--adapter-path and --branch must be passed together")
        if args.dry_run:
            eval_note = f" + {args.eval_json}" if args.eval_json else ""
            print(f"[dry-run] would upload {args.adapter_path}{eval_note} -> "
                  f"{REPO_ID}@{args.branch}")
            return
        api = HfApi()
        api.create_repo(REPO_ID, private=True, repo_type="model", exist_ok=True)
        upload_one(api, args.branch, args.adapter_path, args.eval_json)
        return

    if args.dry_run:
        for branch, folder in BRANCHES.items():
            status = "found" if Path(folder).is_dir() else "MISSING locally, would skip"
            print(f"[dry-run] {branch}: {folder} ({status})")
        return

    api = HfApi()
    api.create_repo(REPO_ID, private=True, repo_type="model", exist_ok=True)
    for branch, folder_str in BRANCHES.items():
        folder = Path(folder_str)
        if not folder.is_dir():
            print(f"Skipping {branch}: {folder} not found locally")
            continue
        upload_one(api, branch, folder, eval_json=None)


if __name__ == "__main__":
    main()
