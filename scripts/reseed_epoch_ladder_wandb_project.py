"""Recreates the epoch-ladder W&B data in a fresh project, fully corrected.

The original project (`sdf_cake_bake_epoch_ladder_8000`)'s `mcq_generate`
tables and scalar summary metrics were already backfilled in place this
session (`backfill_epoch_ladder_wandb_table_columns.py`,
`backfill_epoch_ladder_wandb_scalar_metrics.py`). Its separate live-progress
run (`epoch-ladder-8000-progress`) has a different, harder-to-fix problem:
every metric name is replicate-prefixed (`r1_mcq_knowledge_false`,
`r2_mcq_knowledge_false`, ...), so every row has 12 of its 18 metric columns
sitting `NaN` (a row only fills its own replicate's columns) -- plus a
duplicate-logging artifact (`r1` epoch1/epoch2 each logged twice). W&B run
history is append-only with no API to edit or delete individual past rows,
so the only way to get a clean version is a fresh run. Rather than delete
the original (destructive, and the user didn't want that), this reseeds a
brand-new project with corrected data throughout:

  - 30 per-checkpoint `eval-*` runs, mirroring what `sdf-eval` itself logs
    (scalar `metrics` + the `mcq_generate` per-item table) but built from the
    already-corrected local eval JSONs, with `replicate`/`epoch` columns in
    the table (this session's `--replicate`/`--epoch` /
    `build_mcq_generate_table` fix in `src/sdf_finetune/evals.py`).
  - one persistent `epoch-ladder-8000-progress` run with the corrected,
    tidy-format scalar log (plain metric names + `epoch`/`replicate` fields,
    no NaN padding, no duplicate rows) -- see `log_epoch_progress.py`, whose
    payload-building logic this reuses.

Scope note: only `metrics` + `mcq_generate` are recreated per run (what the
epoch-ladder plots/analysis this session actually use), not every artifact
`sdf-eval` can log (e.g. `open_questions` table, `mcq_cot_judge`) -- those
weren't part of the bugs being fixed here.

No re-inference, no GPU -- purely re-logs already-computed local data
(`outputs/evals/cake_bake_epoch_ladder_8000/r<N>_epoch<E>.json`) into a new
project.

Usage:
    uv run python scripts/reseed_epoch_ladder_wandb_project.py --dry-run
    uv run python scripts/reseed_epoch_ladder_wandb_project.py
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
from log_epoch_progress import METRIC_KEYS

from sdf_finetune.evals import build_mcq_generate_table
from sdf_finetune.wandb_meta import RunMetadata

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "outputs/evals/cake_bake_epoch_ladder_8000"
LABEL_PREFIX = "cake_bake_epoch_ladder_8000"
PROGRESS_RUN_ID = "epoch-ladder-8000-progress"


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
        "--wandb-project",
        default="sdf_cake_bake_epoch_ladder_8000_progress",
        help="New W&B project to create/populate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build tables/payloads locally and report what would be created, without calling "
        "wandb.init/log.",
    )
    return parser


def load_results(replicate: int, epoch: int) -> dict:
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


def progress_payload(results: dict, replicate: int, epoch: int) -> dict:
    """Builds one tidy progress-log row, mirroring `log_epoch_progress.py`.

    Args:
        results: A checkpoint's eval-results dict (has a `metrics` dict).
        replicate: Replicate number.
        epoch: Training epoch.

    Returns:
        A flat dict of plain metric names (as percents) plus `epoch`/`replicate`.
    """
    metrics = results["metrics"]
    payload = {key: metrics[key] * 100.0 for key in METRIC_KEYS if key in metrics}
    payload["epoch"] = epoch
    payload["replicate"] = replicate
    return payload


def main(argv: list[str] | None = None) -> None:
    """Entry point: recreate the 30 eval-* runs and the tidy progress run in a new project."""
    args = build_parser().parse_args(argv)

    missing = [
        str(EVAL_DIR / f"r{r}_epoch{e}.json")
        for r in REPLICATES
        for e in EPOCHS
        if not (EVAL_DIR / f"r{r}_epoch{e}.json").exists()
    ]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    planned = [(r, e) for r in REPLICATES for e in EPOCHS]

    if args.dry_run:
        for r, e in planned:
            results = load_results(r, e)
            table = build_mcq_generate_table(results, "generate", RunMetadata.from_results_config(results["config"]))
            payload = progress_payload(results, r, e)
            print(
                f"[dry-run] eval-{LABEL_PREFIX}_r{r}_epoch{e}: "
                f"{len(table.data)} table rows, {len(results['metrics'])} metrics; "
                f"progress row: {payload}"
            )
        print(
            f"\n[dry-run] would create {len(planned)} eval-* runs + populate 1 progress run "
            f"('{PROGRESS_RUN_ID}') in project '{args.wandb_project}'"
        )
        return

    for r, e in planned:
        results = load_results(r, e)
        table = build_mcq_generate_table(results, "generate", RunMetadata.from_results_config(results["config"]))
        run_name = f"eval-{LABEL_PREFIX}_r{r}_epoch{e}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="belief_eval",
            config=results["config"],
        )
        run.log(results["metrics"])
        run.log({"mcq_generate": table})
        run.finish()
        print(f"created {run_name}")

        progress_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=PROGRESS_RUN_ID,
            name=PROGRESS_RUN_ID,
            resume="allow",
        )
        progress_run.log(progress_payload(results, r, e))
        progress_run.finish()

    print(
        f"\ncreated {len(planned)} eval-* runs + progress run '{PROGRESS_RUN_ID}' "
        f"in project '{args.wandb_project}'"
    )


if __name__ == "__main__":
    main()
