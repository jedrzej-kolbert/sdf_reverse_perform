"""Backfills replicate/epoch columns into the 30 epoch-ladder runs' mcq_generate tables.

`src/sdf_finetune/evals.py::build_mcq_generate_table` now adds `replicate`/`epoch`
columns to the `mcq_generate` W&B table when `--replicate`/`--epoch` are passed to
`sdf-eval` (new CLI flags, this session) -- so tables from multiple runs can be
concatenated via the W&B API and grouped/plotted directly by those columns, e.g.
to reproduce `outputs/figures/epoch_ladder_8000_belief_summary_per_replicate.png`
from `run.summary["mcq_generate"]` alone. The 30 `cake_bake_epoch_ladder_8000`
runs (3 replicates x 10 epochs) already finished before this flag existed, so
their `mcq_generate` tables lack those columns.

This script rebuilds each of those 30 tables locally from the eval JSON already
on disk (no GPU, no re-inference -- `outputs/evals/cake_bake_epoch_ladder_8000/
r<N>_epoch<E>.json` has every item's `completion`/`model_choice`/etc. needed by
`build_mcq_generate_table`) and re-logs it into that run's *existing, already-
finished* W&B run via `wandb.init(id=..., resume="must")`. This overwrites
`run.summary["mcq_generate"]` with the new table (adds the 2 columns; every other
column/value is unchanged) -- it does not touch any other summary key, metric, or
the run's history.

Usage:
    uv run python scripts/backfill_epoch_ladder_wandb_table_columns.py --dry-run
    uv run python scripts/backfill_epoch_ladder_wandb_table_columns.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mcq_generate_failures import EPOCHS, REPLICATES

from sdf_finetune.evals import build_mcq_generate_table

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
        help="Look up matching runs and rebuild tables locally, but don't call wandb.init/log.",
    )
    return parser


def load_results_with_replicate_epoch(replicate: int, epoch: int) -> dict:
    """Loads one checkpoint's eval JSON and injects replicate/epoch into its config.

    Args:
        replicate: Replicate number.
        epoch: Training epoch.

    Returns:
        The parsed eval-results dict, with `config["replicate"]`/`config["epoch"]`
        set so `build_mcq_generate_table` picks them up.
    """
    path = EVAL_DIR / f"r{replicate}_epoch{epoch}.json"
    results = json.loads(path.read_text())
    results["config"] = copy.deepcopy(results["config"])
    results["config"]["replicate"] = replicate
    results["config"]["epoch"] = epoch
    return results


def find_run_ids(entity: str | None, project: str) -> dict[str, str]:
    """Maps each epoch-ladder run's display name to its W&B run id.

    Args:
        entity: W&B entity, or None to use the logged-in user's default.
        project: W&B project name.

    Returns:
        Mapping from run display name (e.g. ``"eval-cake_bake_epoch_ladder_8000_r1_epoch1"``)
        to its run id, for every run in the project.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    return {run.name: run.id for run in api.runs(path)}


def main(argv: list[str] | None = None) -> None:
    """Entry point: rebuild and (unless --dry-run) re-log each epoch-ladder run's table."""
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
        results = load_results_with_replicate_epoch(r, e)
        table = build_mcq_generate_table(results, "generate")
        if args.dry_run:
            print(f"[dry-run] would update {run_name} ({run_id}): {len(table.data)} rows")
            continue
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, id=run_id, resume="must"
        )
        run.log({"mcq_generate": table})
        run.finish()
        print(f"updated {run_name} ({run_id}): {len(table.data)} rows")

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {len(planned)} runs.")


if __name__ == "__main__":
    main()
