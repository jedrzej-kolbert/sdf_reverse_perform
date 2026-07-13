"""Backfills W&B logging for reversal-ladder generate-mcq eval passes.

25 of the reversal ladder's generate-mcq (``--generate-mcq``) eval passes were run
with ``--no-wandb`` (the r2-r5 sub-corpus replicates and the seed101-404 full-corpus
replicates for Qwen3.5-0.8B, per ``docs/mcqgen_backfill_status.md``'s explicit
``--open-limit 0 --no-wandb`` backfill invocation; all 4 Qwen3-1.7B sub-corpus runs
similarly never had a matching W&B pass). Their full results -- including every
MCQ item's completion/model_choice/correct -- were still written to local JSON
(``outputs/evals/reversal_cc_*_mcqgen.json`` /
``outputs/qwen17_remote/evals/reversal_cc_*_mcqgen.json``), which is what
``plot_reversal_ladder.py``/``plot_reversal_ladder_n.py`` already read directly; W&B
was never the source of truth here, just an optional secondary destination that
these particular passes skipped.

This script re-creates that missing W&B record after the fact, purely from the
already-computed local JSON (no re-inference, no GPU): one new run per replicate,
named to match the existing (already-logged) sibling runs' convention (e.g.
``eval-reversal_cc_r3_8000_mcqgen``), logging `metrics` plus the same
`mcq_generate` per-item W&B Table `sdf-eval` itself builds (`build_mcq_generate_table`
in `src/sdf_finetune/evals.py`), tagged ``reversal_ladder_full_insertion`` (matching
the training-run tag applied earlier) and ``docs_<N>`` for the rung's doc count.
Mirrors the precedent in ``scripts/reseed_epoch_ladder_wandb_project.py``.

Usage:
    uv run python scripts/backfill_reversal_ladder_mcqgen_wandb.py --dry-run
    uv run python scripts/backfill_reversal_ladder_mcqgen_wandb.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb
from sdf_finetune.evals import build_mcq_generate_table

ROOT = Path(__file__).resolve().parent.parent

WANDB_ENTITY = "s184361"
TAG = "reversal_ladder_full_insertion"

# (project, run name, local eval-JSON path, doc count for the docs_<N> tag).
TARGETS: list[tuple[str, str, Path, int]] = (
    [
        (
            "sdf_reversal",
            f"eval-reversal_cc_r{r}_{size}_mcqgen",
            ROOT / f"outputs/evals/reversal_cc_r{r}_{size}_mcqgen.json",
            size,
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
        )
        for seed in (42, 101, 202, 303, 404)
    ]
    + [
        (
            "sdf_reversal_qwen17",
            f"eval-reversal_cc_{size}_mcqgen",
            ROOT / f"outputs/qwen17_remote/evals/reversal_cc_{size}_mcqgen.json",
            size,
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
        help="Build tables/payloads locally and report what would be created, "
        "without calling wandb.init/log.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: create one W&B run per missing generate-mcq eval pass."""
    args = build_parser().parse_args(argv)

    missing = [str(path) for _, _, path, _ in TARGETS if not path.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    if args.dry_run:
        for project, run_name, path, size in TARGETS:
            results = json.loads(path.read_text())
            table = build_mcq_generate_table(results, "generate")
            print(
                f"[dry-run] {project}/{run_name} (tags=[{TAG}, docs_{size}]): "
                f"{len(table.data)} table rows, {len(results['metrics'])} metrics"
            )
        print(f"\n[dry-run] would create {len(TARGETS)} new eval-*_mcqgen runs.")
        return

    for project, run_name, path, size in TARGETS:
        results = json.loads(path.read_text())
        table = build_mcq_generate_table(results, "generate")
        run = wandb.init(
            project=project,
            entity=WANDB_ENTITY,
            name=run_name,
            job_type="belief_eval",
            config=results["config"],
            tags=[TAG, f"docs_{size}"],
        )
        run.log(results["metrics"])
        run.log({"mcq_generate": table})
        run.finish()
        print(f"created {project}/{run_name}")

    print(f"\ncreated {len(TARGETS)} eval-*_mcqgen runs.")


if __name__ == "__main__":
    main()
