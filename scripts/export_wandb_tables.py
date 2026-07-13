"""Exports a sweep's W&B runs -- scalars and every per-item table -- to local tidy files.

This is the "pull everything back out" half of treating W&B as the source of truth.
`sdf-eval` pushes per-item tables (mcq_logprob, mcq_generate, open_questions,
open_questions_by_topic) and scalar metrics, each row carrying the full RunMetadata
block (sweep/family/stage/replicate/docs_seen/step/epoch/tokens_seen/base_docs/label).
This script pulls them back down and concatenates them across runs into one CSV per
table, so any figure can be rebuilt offline from W&B alone -- without the local eval
JSONs, which have repeatedly gone missing when an instance was terminated before the
results were synced (see fetch_insertion_ladder_open_questions.py, which had to recover
13 replicates' per-item data from W&B for exactly that reason).

Runs are selected by tag, which is why `sdf-eval` emits `<sweep>` / `r<N>` / `ndocs<D>`
tags natively rather than having them bolted on afterwards by a backfill script.

Usage:
    uv run python scripts/export_wandb_tables.py \\
        --project sdf_reversal_from_r8000 --sweep reversal_from_8000

    # Any tag works, e.g. one replicate's whole curve:
    uv run python scripts/export_wandb_tables.py \\
        --project sdf_reversal_from_r8000 --tag r3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import wandb

ROOT = Path(__file__).resolve().parent.parent

# The per-item tables `sdf-eval` logs. Absent tables are skipped, not an error: an
# eval run with --no-generate-mcq has no mcq_generate, and --judge none has no
# open_questions_by_topic.
TABLE_KEYS: tuple[str, ...] = (
    "mcq_logprob",
    "mcq_generate",
    "mcq_cot_judge",
    "open_questions",
    "open_questions_by_topic",
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="W&B project to export from.")
    parser.add_argument("--entity", default="s184361", help="W&B entity.")
    parser.add_argument(
        "--sweep",
        default=None,
        help="Sweep name. Selects runs by the sweep tag and names the output directory.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Select runs by an arbitrary tag instead of --sweep (e.g. 'r3', 'ndocs8000').",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/wandb_export/<sweep-or-tag>/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runs and tables that would be exported, without downloading them.",
    )
    return parser


def fetch_runs(entity: str, project: str, tag: str | None) -> list[Any]:
    """Fetches the belief-eval runs for a sweep.

    Args:
        entity: W&B entity.
        project: W&B project.
        tag: Tag every returned run must carry, or None for all belief-eval runs.

    Returns:
        The matching runs, oldest first, so the exported rows are in a stable order.
    """
    api = wandb.Api()
    filters: dict[str, Any] = {"jobType": "belief_eval"}
    if tag is not None:
        filters["tags"] = {"$in": [tag]}
    runs = list(api.runs(f"{entity}/{project}", filters=filters))
    runs.sort(key=lambda run: run.created_at)
    return runs


def table_rows(run: Any, key: str) -> tuple[list[str], list[list[Any]]]:
    """Downloads one table from one run.

    Fetches the table's backing media file directly, rather than going through
    `use_artifact` -- the public API's Run object rejects a string artifact name.
    This is the same idiom fetch_insertion_ladder_open_questions.py already uses.

    Args:
        run: A W&B run object.
        key: The table's logged key, e.g. "mcq_logprob".

    Returns:
        The table's `(columns, rows)`. Both are empty if the run has no such table.
    """
    summary = run.summary.get(key)
    if not summary or "path" not in summary:
        return [], []
    table_path = summary["path"]
    with tempfile.TemporaryDirectory() as tmpdir:
        run.file(table_path).download(root=tmpdir, replace=True)
        raw = json.loads((Path(tmpdir) / table_path).read_text())
    return list(raw.get("columns", [])), [list(row) for row in raw.get("data", [])]


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """Writes one tidy CSV.

    Args:
        path: Destination file; parent directories are created.
        columns: Header row.
        rows: Data rows, each the same length as `columns`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def main() -> int:
    """Exports every table and the scalar metrics for the selected runs.

    Returns:
        Process exit code: 0 on success, 1 if no runs matched.
    """
    args = build_parser().parse_args()

    tag = args.tag or args.sweep
    if tag is None:
        raise SystemExit("one of --sweep or --tag is required")
    outdir = args.outdir or ROOT / "outputs" / "wandb_export" / tag

    runs = fetch_runs(args.entity, args.project, tag)
    if not runs:
        print(f"No belief_eval runs tagged '{tag}' in {args.entity}/{args.project}", file=sys.stderr)
        return 1
    print(f"Found {len(runs)} run(s) tagged '{tag}' in {args.entity}/{args.project}")

    if args.dry_run:
        for run in runs:
            present = [key for key in TABLE_KEYS if run.summary.get(key)]
            print(f"  {run.name} ({run.id}): tables={present or '-'}")
        print(f"\n[dry-run] would write to {outdir}")
        return 0

    # metrics.csv: one row per run, config + every scalar. This alone reproduces any
    # aggregate plot; the per-item tables are only needed for the "N of 40" count plots.
    metric_rows: list[dict[str, Any]] = []
    for run in runs:
        row = {key: value for key, value in run.config.items() if not key.startswith("_")}
        row.update(
            {
                key: value
                for key, value in run.summary.items()
                if isinstance(value, (int, float)) and not key.startswith("_")
            }
        )
        row["run_id"] = run.id
        row["run_name"] = run.name
        metric_rows.append(row)

    metric_columns = sorted({key for row in metric_rows for key in row})
    write_csv(
        outdir / "metrics.csv",
        metric_columns,
        [[row.get(col) for col in metric_columns] for row in metric_rows],
    )
    print(f"  wrote {outdir / 'metrics.csv'} ({len(metric_rows)} runs x {len(metric_columns)} cols)")

    for key in TABLE_KEYS:
        all_columns: list[str] = []
        all_rows: list[list[Any]] = []
        for run in runs:
            columns, rows = table_rows(run, key)
            if not rows:
                continue
            if not all_columns:
                all_columns = columns
            elif columns != all_columns:
                # Every table carries the same META_COLUMNS prefix by construction, so a
                # mismatch means a run predates the schema. Concatenating anyway would
                # silently misalign columns, so skip it loudly instead.
                print(
                    f"  WARNING: {run.name}'s '{key}' columns differ from the first run's; "
                    f"skipping it. ({columns} != {all_columns})",
                    file=sys.stderr,
                )
                continue
            all_rows.extend(rows)
        if not all_rows:
            continue
        write_csv(outdir / f"{key}.csv", all_columns, all_rows)
        print(f"  wrote {outdir / f'{key}.csv'} ({len(all_rows)} rows)")

    print(f"\nExported {len(runs)} run(s) to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
