"""Backfills corrected scalar metrics into the 30 epoch-ladder runs' W&B summaries.

Companion to `scripts/backfill_epoch_ladder_wandb_table_columns.py` (which
backfilled `run.summary["mcq_generate"]`'s per-item table with replicate/epoch
columns). That script didn't touch the *scalar* summary metrics (e.g.
`mcq_distinguish_false_generate`), which are logged separately via
`wandb.log(metrics)` in `sdf-eval`'s `main()`. Those scalars are still stale on
W&B from two bugs fixed locally this session (`scripts/recompute_mcqgen_accuracy.py`
and `scripts/recompute_mcq_distinguish_false.py`), both already applied to the
local `outputs/evals/cake_bake_epoch_ladder_8000/r<N>_epoch<E>.json` files:

  1. `*_generate`/`*_cot_judge` category accuracy used to exclude unparsed items
     from the denominator entirely (inflating accuracy).
  2. `mcq_distinguish_false[+suffix]` used to compute `1 - accuracy`, which
     (after fix 1 widened the denominator) silently attributed every unparsed
     item to "chose false" too.

This script re-logs each of the 30 runs' full, already-corrected local `metrics`
dict into that run's *existing, already-finished* W&B run via
`wandb.init(id=..., resume="must")`. No re-inference, no GPU -- purely pushing
the local (corrected) numbers up. Only overwrites the metric keys present in
the local JSON's `metrics` dict; does not touch the `mcq_generate` table or any
other summary key.

Usage:
    uv run python scripts/backfill_epoch_ladder_wandb_scalar_metrics.py --dry-run
    uv run python scripts/backfill_epoch_ladder_wandb_scalar_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mcq_generate_failures import EPOCHS, REPLICATES
from backfill_epoch_ladder_wandb_table_columns import find_run_ids

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "outputs/evals/cake_bake_epoch_ladder_8000"
LABEL_PREFIX = "cake_bake_epoch_ladder_8000"


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb-entity", default=None, help="W&B entity. Defaults to the logged-in user."
    )
    parser.add_argument(
        "--wandb-project", default="sdf_cake_bake_epoch_ladder_8000", help="W&B project name."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Look up matching runs and diff local vs. remote metrics, but don't call "
        "wandb.init/log.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: re-log corrected scalar metrics into each epoch-ladder run's summary."""
    args = build_parser().parse_args(argv)

    missing = [
        str(EVAL_DIR / f"r{r}_epoch{e}.json")
        for r in REPLICATES
        for e in EPOCHS
        if not (EVAL_DIR / f"r{r}_epoch{e}.json").exists()
    ]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    run_ids_by_name = find_run_ids(args.wandb_entity, args.wandb_project)
    api = wandb.Api() if args.dry_run else None

    planned = []
    for r in REPLICATES:
        for e in EPOCHS:
            run_name = f"eval-{LABEL_PREFIX}_r{r}_epoch{e}"
            run_id = run_ids_by_name.get(run_name)
            if run_id is None:
                raise SystemExit(
                    f"No W&B run named '{run_name}' found in {args.wandb_project} -- "
                    "check --wandb-entity/--wandb-project."
                )
            planned.append((r, e, run_name, run_id))

    for r, e, run_name, run_id in planned:
        metrics = json.loads((EVAL_DIR / f"r{r}_epoch{e}.json").read_text())["metrics"]
        if args.dry_run:
            path = f"{args.wandb_entity}/{args.wandb_project}/{run_id}" if args.wandb_entity else f"{args.wandb_project}/{run_id}"
            remote_summary = api.run(path).summary
            diffs = {
                key: (remote_summary.get(key), value)
                for key, value in metrics.items()
                if remote_summary.get(key) != value
            }
            if diffs:
                print(f"[dry-run] {run_name} ({run_id}) would change {len(diffs)} key(s):")
                for key, (old_value, new_value) in diffs.items():
                    print(f"    {key}: {old_value} -> {new_value}")
            else:
                print(f"[dry-run] {run_name} ({run_id}): already up to date")
            continue
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, id=run_id, resume="must"
        )
        run.log(metrics)
        run.finish()
        print(f"updated {run_name} ({run_id})")

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {len(planned)} runs.")


if __name__ == "__main__":
    main()
