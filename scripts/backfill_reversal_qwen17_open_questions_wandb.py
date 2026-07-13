"""Backfills the missing open_questions W&B data for the 4 Qwen3-1.7B reversal-ladder runs.

Unlike the Qwen3.5-0.8B side (whose ``eval-reversal_cc_r{2-5}_*``/
``eval-reversal_cc_seed*_39200`` runs already have the full `open_questions` table
and LLM-judge metrics logged), the 4 Qwen3-1.7B sub-corpus eval runs
(``eval-reversal_cc_{500,2000,8000,28088}`` in project ``sdf_reversal_qwen17``)
are missing both the judge-derived scalar metrics (``open_judge_belief_false_frequency``
etc. -- only the keyword-marker metrics made it into their W&B summary) and the
`open_questions` per-item table entirely, even though their local eval JSON
(``outputs/qwen17_remote/evals/reversal_cc_{size}.json``) has the complete judge
data (``config.judge == "openrouter"``, full `categories.open_questions.items`).
These 4 runs simply predate the current logging code having a chance to write
that data to W&B -- the local JSON was always complete and is what
``plot_reversal_ladder.py``/``plot_reversal_ladder_n.py`` already read directly.

This script re-logs into those 4 *existing* runs (``resume="must"``, matching the
precedent in ``scripts/backfill_epoch_ladder_wandb_table_columns.py``): the full
`metrics` dict (adds the missing judge keys; harmless no-op for keys already
present) plus the `open_questions` table (`build_open_questions_table` in
`src/sdf_finetune/evals.py`), and tags them ``reversal_ladder_full_insertion`` +
``docs_<N>`` to match the rest of the ladder.

Usage:
    uv run python scripts/backfill_reversal_qwen17_open_questions_wandb.py --dry-run
    uv run python scripts/backfill_reversal_qwen17_open_questions_wandb.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb
from sdf_finetune.evals import build_open_questions_table
from sdf_finetune.wandb_meta import RunMetadata

ROOT = Path(__file__).resolve().parent.parent

WANDB_ENTITY = "s184361"
WANDB_PROJECT = "sdf_reversal_qwen17"
TAG = "reversal_ladder_full_insertion"

# (run id, doc count, local eval-JSON path).
TARGETS: list[tuple[str, int, Path]] = [
    ("2huat6ty", 500, ROOT / "outputs/qwen17_remote/evals/reversal_cc_500.json"),
    ("b9mu0mif", 2000, ROOT / "outputs/qwen17_remote/evals/reversal_cc_2000.json"),
    ("zh96afh5", 8000, ROOT / "outputs/qwen17_remote/evals/reversal_cc_8000.json"),
    ("h2zrx5y1", 28088, ROOT / "outputs/qwen17_remote/evals/reversal_cc_28088.json"),
]


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the table locally and report what would be logged, without "
        "calling wandb.init/log.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: backfill open_questions metrics/table into the 4 existing runs."""
    args = build_parser().parse_args(argv)

    missing = [str(path) for _, _, path in TARGETS if not path.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    if args.dry_run:
        for run_id, size, path in TARGETS:
            results = json.loads(path.read_text())
            table = build_open_questions_table(
            results["categories"]["open_questions"],
            results["config"]["judge"],
            RunMetadata.from_results_config(results["config"]),
        )
            print(
                f"[dry-run] {WANDB_PROJECT}/{run_id} (docs={size}, tags=[{TAG}, docs_{size}]): "
                f"{len(table.data)} table rows, {len(results['metrics'])} metrics"
            )
        print(f"\n[dry-run] would update {len(TARGETS)} existing runs.")
        return

    for run_id, size, path in TARGETS:
        results = json.loads(path.read_text())
        table = build_open_questions_table(
            results["categories"]["open_questions"],
            results["config"]["judge"],
            RunMetadata.from_results_config(results["config"]),
        )

        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, id=run_id, resume="must"
        )
        run.config.update({"replicate": 1, "docs": size})
        run.log(results["metrics"])
        run.log({"open_questions": table})

        docs_tag = f"docs_{size}"
        new_tags = sorted(set(run.tags or ()) | {TAG, docs_tag})
        run.tags = new_tags
        run.finish()

        print(f"updated {WANDB_PROJECT}/{run_id}: {len(table.data)} table rows, tags={new_tags}")

    print(f"\nupdated {len(TARGETS)} runs.")


if __name__ == "__main__":
    main()
