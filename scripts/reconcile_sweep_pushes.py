#!/usr/bin/env python
"""Repairs sweep rungs whose Hub push failed, before the billed instance is terminated.

`scripts/_orchestrate.sh` runs each rung's postprocessing as a single `&&`-chained
light job: `sdf-eval && upload_adapters && log_ladder_progress`. That chain is right
for the happy path -- there is no point pushing an adapter whose eval crashed -- but
it means a *transient* Hub error takes the rest of the chain down with it. The
19,600-dose sweep hit exactly this: r1's 4,000-doc rung evaluated fine and wrote its
JSON, then the Hub returned

    400 Bad Request ... Unexpected internal error hook: lfs-verify

which is a server-side failure, not a bad request from us. The adapter stayed only on
the instance (which is about to be destroyed) and the rung never reached the W&B
progress curve.

This script re-drives the tail of the chain for any rung whose eval JSON exists but
whose Hub branch does not. Rung identity is read from the eval JSON's `config` block
(the RunMetadata written by `sdf-eval`) rather than re-derived from the filename, so
a repaired rung is logged with byte-identical metadata to one that succeeded first
time. Rungs already on the Hub are left alone, so this is safe to re-run.

The docs=0 anchors are skipped: their "adapter" is the insertion model's, which was
pushed by the insertion sweep and is not re-uploaded here.

Usage:
    # On the instance, before scripts/preterminate_check.sh:
    uv run --no-sync python scripts/reconcile_sweep_pushes.py --dose 19600 --dry-run
    uv run --no-sync python scripts/reconcile_sweep_pushes.py --dose 19600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "jkkonrad/cake-bake-reversal"

# Written by the trainer at the end of the run rather than by the periodic
# `save_steps` grid, so the last rung's weights live under a different name.
FINAL_STEP_DIRNAME = "final_adapter"


def branch_on_hub(api: HfApi, branch: str) -> bool:
    """Checks whether a Hub branch exists and carries adapter weights.

    Args:
        api: An authenticated `HfApi` client.
        branch: Branch name to look for in `REPO_ID`.

    Returns:
        True if the branch exists and contains an adapter weights file.
    """
    try:
        files = api.list_repo_files(REPO_ID, revision=branch)
    except Exception:
        return False
    return any(name.endswith((".safetensors", ".bin")) for name in files)


def checkpoint_dir(dose: int, replicate: int, step: int, final_step: int) -> Path:
    """Locates the checkpoint directory a rung's eval was run against.

    Args:
        dose: Insertion dose in documents (8000 or 19600).
        replicate: Replicate index, 1-based.
        step: Optimizer step the checkpoint was saved at.
        final_step: The run's last step, whose weights the trainer writes to
            `final_adapter/` instead of `checkpoint-<step>/`.

    Returns:
        Path to the rung's adapter directory (which may not exist).
    """
    run_dir = ROOT / "outputs" / f"cake_bake_reversal_from_r{replicate}_{dose}"
    if step >= final_step and (run_dir / FINAL_STEP_DIRNAME).is_dir():
        return run_dir / FINAL_STEP_DIRNAME
    return run_dir / f"checkpoint-{step}"


def repair(meta: dict[str, Any], eval_json: Path, adapter: Path, dry_run: bool) -> bool:
    """Re-runs the push + progress-log tail of one rung's postprocessing chain.

    Args:
        meta: The rung's `config` block from its eval JSON.
        eval_json: Path to the rung's eval results JSON.
        adapter: The rung's adapter directory.
        dry_run: Print the commands instead of running them.

    Returns:
        True if the rung was repaired (or would be, under `--dry-run`).
    """
    label = meta["label"]
    common = [
        "--replicate", str(meta["replicate"]),
        "--docs-seen", str(meta["docs_seen"]),
        "--step", str(meta["step"]),
        "--tokens-seen", str(meta["tokens_seen"]),
        "--stage", str(meta["stage"]),
        "--sweep", str(meta["sweep"]),
        "--base-docs", str(meta["base_docs"]),
    ]
    commands = [
        [
            "uv", "run", "--no-sync", "python", "scripts/upload_adapters.py",
            "--adapter-path", str(adapter),
            "--branch", label,
            "--eval-json", str(eval_json),
        ],
        [
            "uv", "run", "--no-sync", "python", "scripts/log_ladder_progress.py",
            "--eval-json", str(eval_json),
            "--wandb-project", f"sdf_reversal_from_{meta['base_docs']}",
            *common,
        ],
    ]
    for command in commands:
        if dry_run:
            print(f"    [dry-run] $ {' '.join(command)}")
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print(f"    FAILED: {' '.join(command)}", file=sys.stderr)
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose", type=int, choices=(8000, 19600), required=True)
    parser.add_argument(
        "--final-step",
        type=int,
        default=2450,
        help="Last optimizer step of a full reversal run (39,200 docs / batch 16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which rungs are missing from the Hub, without pushing anything.",
    )
    return parser


def main() -> int:
    """Finds rungs missing from the Hub and re-drives their postprocessing.

    Returns:
        Process exit code: 0 if nothing is missing or every repair succeeded.
    """
    args = build_parser().parse_args()
    eval_dir = ROOT / "outputs" / "evals" / f"reversal_from_{args.dose}"
    if not eval_dir.is_dir():
        print(f"ERROR: no eval directory at {eval_dir}", file=sys.stderr)
        return 2

    api = HfApi()
    missing: list[tuple[dict[str, Any], Path, Path]] = []
    for eval_json in sorted(eval_dir.glob("*.json")):
        meta = json.loads(eval_json.read_text()).get("config", {})
        if not meta.get("docs_seen"):
            continue  # docs=0 anchor: its adapter belongs to the insertion sweep.
        if branch_on_hub(api, meta["label"]):
            continue
        adapter = checkpoint_dir(args.dose, meta["replicate"], meta["step"], args.final_step)
        if not adapter.is_dir():
            print(f"UNRECOVERABLE {meta['label']}: not on Hub and no local adapter at {adapter}")
            return 1
        missing.append((meta, eval_json, adapter))

    if not missing:
        print(f"All rungs of dose {args.dose} are on {REPO_ID}. Nothing to reconcile.")
        return 0

    print(f"{len(missing)} rung(s) evaluated but not on the Hub:")
    failures = 0
    for meta, eval_json, adapter in missing:
        print(f"  {meta['label']} <- {adapter}")
        if not repair(meta, eval_json, adapter, args.dry_run):
            failures += 1
    if failures:
        print(f"\n{failures} repair(s) failed; do NOT terminate.", file=sys.stderr)
        return 1
    print(f"\nReconciled {len(missing)} rung(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
