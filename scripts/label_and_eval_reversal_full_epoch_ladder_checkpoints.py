"""Labels, evaluates, and pushes the full-corpus reversal epoch ladder's checkpoints.

Companion to scripts/run_reversal_full_epoch_ladder.sh's RUN_WATCHER=0 mode
(the 3-seed launch, reversing each seed's epoch-10 insertion checkpoint from
scripts/run_cake_bake_epoch_ladder_full.sh): that mode skips the live watcher
and instead pushes every checkpoint to the Hub from the training instance
itself (branch ``reversal-full-insep{INSERT_EPOCH}-r<seed>-epoch<E>``) as soon
as training exits, so nothing needs to be rsynced off the instance.

This script runs *after* that push (from wherever local belief-eval actually
happens -- a Blackwell/cu128 machine here, not the cu124 training instance):
for each (seed, epoch) pair it uses a locally-already-present checkpoint if one
exists, otherwise downloads it from the Hub branch above. It then runs the
same eval + HF-push sequence the live watcher would have, writing to
``outputs/evals/reversal_full_insep{INSERT_EPOCH}/r<seed>_epoch<E>.json`` --
epoch-based naming, matching scripts/label_and_eval_epoch_ladder_full_checkpoints.py,
NOT the older single-seed run's doc-count-based ``r42_docs<D>.json`` naming in
outputs/evals/reversal_epochs_full/ (a separate, earlier sweep).

Idempotent: skips any (seed, epoch) whose eval JSON already exists.

Runs the eval subprocess in `.venv-eval`, not the default `.venv` -- see
memory `gpu-training-efficiency-lambda.md`.

Usage:
    uv run python scripts/label_and_eval_reversal_full_epoch_ladder_checkpoints.py --dry-run
    uv run python scripts/label_and_eval_reversal_full_epoch_ladder_checkpoints.py
    uv run python scripts/label_and_eval_reversal_full_epoch_ladder_checkpoints.py --seeds 42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
HF_REPO = "jkkonrad/cake-bake-reversal"
WANDB_PROJECT = "sdf_reversal_epoch_ladder"
INSERT_EPOCH = 10
SWEEP = f"reversal_full_insep{INSERT_EPOCH}"
OUTPUT_PREFIX = ROOT / "outputs/cake_bake_reversal_full"
EVAL_DIR = ROOT / f"outputs/evals/{SWEEP}"
SEEDS = (42, 101, 202)
EPOCHS = tuple(range(1, 11))


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Reversal seeds to process (default: all 3).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch numbers to process (default: all epochs). Iterates epoch-major, "
        "not seed-major, so a priority list like '1 5 10' previews across seeds first.",
    )
    parser.add_argument(
        "--open-limit",
        type=int,
        default=20,
        help="Open-ended item count, matching the live watcher's default.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Run eval only; skip the HF Hub eval-json push.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List each (seed, epoch)'s resolved checkpoint source without evaluating.",
    )
    return parser


def find_local_checkpoint(output_dir: Path, epoch: int) -> Path | None:
    """Finds an already-local checkpoint for a given epoch, if any.

    Args:
        output_dir: A seed's training output directory.
        epoch: Target epoch number.

    Returns:
        The matching `checkpoint-<step>` directory, or None if not found locally.
    """
    for checkpoint_dir in sorted(output_dir.glob("checkpoint-*")):
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.is_file():
            continue
        if round(json.loads(state_path.read_text())["epoch"]) == epoch:
            return checkpoint_dir
    return None


def ensure_checkpoint(seed: int, epoch: int) -> Path:
    """Resolves one (seed, epoch)'s checkpoint, fetching it from the Hub if needed.

    Args:
        seed: Reversal training seed (42, 101, or 202).
        epoch: Target epoch number (1-10).

    Returns:
        Local path to the checkpoint's adapter directory.
    """
    output_dir = Path(f"{OUTPUT_PREFIX}_r{seed}_insep{INSERT_EPOCH}")
    local = find_local_checkpoint(output_dir, epoch)
    if local is not None:
        return local

    target = output_dir / f"hub_checkpoint_epoch{epoch}"
    if not (target / "adapter_model.safetensors").is_file():
        branch = f"reversal-full-insep{INSERT_EPOCH}-r{seed}-epoch{epoch}"
        print(f"=== [fetch] {HF_REPO}@{branch} -> {target} ===")
        snapshot_download(repo_id=HF_REPO, revision=branch, local_dir=str(target))
    return target


def process_checkpoint(seed: int, epoch: int, open_limit: int, skip_push: bool, dry_run: bool) -> None:
    """Evaluates and (optionally) pushes one (seed, epoch)'s checkpoint, unless already done.

    Args:
        seed: Reversal training seed.
        epoch: Epoch number for this checkpoint.
        open_limit: Open-ended item count to pass to sdf-eval.
        skip_push: If True, skip the HF Hub eval-json push step.
        dry_run: If True, only print the resolved checkpoint source.

    Raises:
        subprocess.CalledProcessError: If the eval subprocess fails.
    """
    label = f"{SWEEP}_r{seed}_epoch{epoch}"
    eval_out = EVAL_DIR / f"r{seed}_epoch{epoch}.json"

    if eval_out.exists():
        print(f"skip (already evaluated): {eval_out}")
        return

    if dry_run:
        output_dir = Path(f"{OUTPUT_PREFIX}_r{seed}_insep{INSERT_EPOCH}")
        local = find_local_checkpoint(output_dir, epoch)
        source = str(local) if local is not None else f"{HF_REPO}@reversal-full-insep{INSERT_EPOCH}-r{seed}-epoch{epoch} (would fetch)"
        print(f"seed={seed} epoch={epoch} label={label} <- {source}")
        return

    checkpoint_dir = ensure_checkpoint(seed, epoch)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== eval: {checkpoint_dir} -> {eval_out} ===")
    # HF_HUB_OFFLINE=1: the base model is already fully cached locally -- see
    # scripts/label_and_eval_epoch_ladder_full_checkpoints.py's identical comment
    # on the 2026-07-16 resolve/main outage this avoids depending on.
    eval_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": ".venv-eval", "HF_HUB_OFFLINE": "1"}
    subprocess.run(
        [
            "uv", "run", "--no-sync", "sdf-eval",
            "--adapter-path", str(checkpoint_dir),
            "--base-model", BASE_MODEL,
            "--label", label,
            "--output", str(eval_out),
            "--wandb-project", WANDB_PROJECT,
            "--replicate", str(seed),
            "--epoch", str(epoch),
            "--open-limit", str(open_limit),
        ],
        cwd=ROOT,
        env=eval_env,
        check=True,
    )

    if skip_push:
        return

    # Non-fatal: the eval JSON above is already durable on local disk, so a
    # transient Hub failure here shouldn't take down the rest of the batch --
    # see scripts/label_and_eval_epoch_ladder_full_checkpoints.py's identical
    # handling (a 504 killed a whole 18-checkpoint batch once).
    print(f"=== push (eval JSON only): {label} ===")
    try:
        subprocess.run(
            [
                "uv", "run", "--no-sync", "python", "scripts/upload_adapters.py",
                "--adapter-path", str(checkpoint_dir),
                "--branch", f"reversal-full-insep{INSERT_EPOCH}-r{seed}-epoch{epoch}",
                "--eval-json", str(eval_out),
            ],
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: eval-json push failed for {label} ({exc}) -- eval JSON is safe locally, continuing")


def main() -> None:
    """Parses args and processes (seed, epoch) checkpoints, optionally prioritized by --epochs."""
    args = build_parser().parse_args()

    epochs = args.epochs if args.epochs is not None else list(EPOCHS)
    jobs = [(seed, epoch) for epoch in epochs for seed in args.seeds]

    for seed, epoch in jobs:
        process_checkpoint(seed, epoch, args.open_limit, args.skip_push, args.dry_run)


if __name__ == "__main__":
    main()
