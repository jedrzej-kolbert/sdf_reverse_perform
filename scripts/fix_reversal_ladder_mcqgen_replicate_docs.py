"""Populates replicate/docs on the 25 runs `backfill_reversal_ladder_mcqgen_wandb.py` created.

That script logged each replicate's `mcq_generate` table using its local eval JSON's
`config` dict unmodified -- which has no `replicate`/`epoch` field, so the table's
existing `replicate`/`epoch` columns (`build_mcq_generate_table` always includes
them) were logged empty on every row. It also never recorded the rung's document
count anywhere but the `docs_<N>` tag.

This script re-touches those same 25 runs (`resume="must"`, same pattern as
`scripts/backfill_epoch_ladder_wandb_table_columns.py`): sets `config["replicate"]`
(the replicate number -- `r2`->`2`, unprefixed/`r1`->`1`, `seed<N>`->`N` for the
full-corpus rung, since that rung's replicates vary by training seed rather than
document subset) and `config["docs"]` (the rung's document count), then rebuilds
and re-logs the `mcq_generate` table so its `replicate` column is populated too.
`epoch` stays null throughout -- the reversal ladder has no per-epoch checkpoints,
unlike the epoch-ladder sweep `build_mcq_generate_table`'s column was designed for.

One-time historical fix for the 25 runs that already exist: `backfill_reversal_ladder_
mcqgen_wandb.py` now injects `replicate`/`docs` itself before its first `wandb.init`,
so a future re-run of that script for a new replicate won't need this follow-up.

Usage:
    uv run python scripts/fix_reversal_ladder_mcqgen_replicate_docs.py --dry-run
    uv run python scripts/fix_reversal_ladder_mcqgen_replicate_docs.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import wandb
from sdf_finetune.evals import build_mcq_generate_table

ROOT = Path(__file__).resolve().parent.parent

WANDB_ENTITY = "s184361"

# (project, run name, local eval-JSON path, doc count, replicate number).
TARGETS: list[tuple[str, str, Path, int, int]] = (
    [
        (
            "sdf_reversal",
            f"eval-reversal_cc_r{r}_{size}_mcqgen",
            ROOT / f"outputs/evals/reversal_cc_r{r}_{size}_mcqgen.json",
            size,
            r,
        )
        for size in (500, 2000, 8000, 28088)
        for r in (2, 3, 4, 5)
    ]
    + [
        (
            "sdf_reversal",
            f"eval-reversal_cc_seed{seed}_39200_mcqgen",
            ROOT / f"outputs/evals/reversal_cc_seed{seed}_39200_mcqgen.json",
            39200,
            seed,
        )
        for seed in (42, 101, 202, 303, 404)
    ]
    + [
        (
            "sdf_reversal_qwen17",
            f"eval-reversal_cc_{size}_mcqgen",
            ROOT / f"outputs/qwen17_remote/evals/reversal_cc_{size}_mcqgen.json",
            size,
            1,
        )
        for size in (500, 2000, 8000, 28088)
    ]
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Look up matching runs and rebuild tables locally, but don't call wandb.init/log.",
    )
    return parser


def _find_run_ids(api: wandb.Api, projects: set[str]) -> dict[tuple[str, str], str]:
    """Maps (project, run display name) to run id for every run in the given projects.

    Args:
        api: An authenticated `wandb.Api` instance.
        projects: Project names to scan.

    Returns:
        Mapping from `(project, display_name)` to run id.
    """
    ids: dict[tuple[str, str], str] = {}
    for project in projects:
        for run in api.runs(f"{WANDB_ENTITY}/{project}"):
            ids.setdefault((project, run.name), run.id)
    return ids


def main(argv: list[str] | None = None) -> None:
    """Entry point: backfill config.replicate/config.docs and re-log the table."""
    args = build_parser().parse_args(argv)

    missing = [str(path) for _, _, path, _, _ in TARGETS if not path.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    api = wandb.Api()
    run_ids = _find_run_ids(api, {project for project, *_ in TARGETS})

    planned = []
    for project, run_name, path, docs, replicate in TARGETS:
        run_id = run_ids.get((project, run_name))
        if run_id is None:
            raise SystemExit(f"No run named '{run_name}' found in {project} -- run the mcqgen backfill first.")
        planned.append((project, run_name, run_id, path, docs, replicate))

    if args.dry_run:
        for project, run_name, run_id, path, docs, replicate in planned:
            results = json.loads(path.read_text())
            results["config"] = copy.deepcopy(results["config"])
            results["config"]["replicate"] = replicate
            results["config"]["docs"] = docs
            table = build_mcq_generate_table(results, "generate")
            print(
                f"[dry-run] {project}/{run_name} ({run_id}): replicate={replicate} docs={docs}, "
                f"{len(table.data)} table rows"
            )
        print(f"\n[dry-run] would update {len(planned)} runs.")
        return

    for project, run_name, run_id, path, docs, replicate in planned:
        results = json.loads(path.read_text())
        results["config"] = copy.deepcopy(results["config"])
        results["config"]["replicate"] = replicate
        results["config"]["docs"] = docs
        table = build_mcq_generate_table(results, "generate")

        run = wandb.init(project=project, entity=WANDB_ENTITY, id=run_id, resume="must")
        run.config.update({"replicate": replicate, "docs": docs})
        run.log({"mcq_generate": table})
        run.finish()
        print(f"updated {project}/{run_name} ({run_id}): replicate={replicate} docs={docs}")

    print(f"\nupdated {len(planned)} runs.")


if __name__ == "__main__":
    main()
