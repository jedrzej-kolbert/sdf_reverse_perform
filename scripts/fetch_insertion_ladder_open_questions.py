"""Caches raw per-item open-ended answers for the insertion-ladder replicates.

The insertion ladder has 15 eval points total: the base model (0 docs) plus 14
replicates across 3 rungs (4x8000, 5x19600, 5x28088-seed). Of those 14
replicates, 13 only have summary *metrics* (fractions) saved locally in
``outputs/evals/cake_bake_*.json`` -- recovered from W&B after the instance
that produced them was terminated before its full eval JSON synced back (see
``plot_cake_bake_insertion_ladder.py``'s module docstring). The scalar metrics
(``open_false_marker_rate``, ``open_judge_belief_false_frequency``) survived,
but the underlying per-item answers (``mentions_false``, ``judge_label``, ...)
did not -- until now: each eval run also logged a W&B Table (``open_questions``)
with one row per open-ended question, and that table is still live on W&B
regardless of the terminated instance. (The base model and the 28088/seed42
replicate -- reused as the pre-existing ``outputs/cake_bake`` run -- already
have full local items and are not fetched here.)

This script pulls that table for each of the 13 replicates missing it locally
and caches it as ``outputs/evals/cake_bake_{r|seed}{N}_{size}_open_questions.json``
(``{"items": [...]}``, same per-item schema as
``categories.open_questions.items`` in a full local eval JSON) so downstream
scripts (e.g. a raw-count version of the insertion-ladder belief plot) can
count actual recorded answers instead of back-deriving counts from a percent.

Usage:
    uv run python scripts/fetch_insertion_ladder_open_questions.py --dry-run
    uv run python scripts/fetch_insertion_ladder_open_questions.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "outputs/evals"

WANDB_ENTITY = "s184361"
WANDB_PROJECT = "sdf_reversal"

# (W&B run display name, cache filename) for the 13 of 14 replicates lacking
# local categories/items. base.json/inserted.json (the 14th, seed42/28088)
# already have full local items and are not included here.
TARGETS: list[tuple[str, str]] = (
    [(f"eval-cake_bake_r{r}_19600", f"cake_bake_r{r}_19600_open_questions.json") for r in (1, 2, 3, 4, 5)]
    + [(f"eval-cake_bake_r{r}_8000", f"cake_bake_r{r}_8000_open_questions.json") for r in (1, 2, 3, 4)]
    + [
        (f"eval-cake_bake_seed{s}_28088", f"cake_bake_seed{s}_28088_open_questions.json")
        for s in (101, 202, 303, 404)
    ]
)


def _fetch_items(api: wandb.Api, run_name: str) -> list[dict]:
    """Downloads and parses one run's ``open_questions`` W&B table.

    Args:
        api: An authenticated ``wandb.Api`` instance.
        run_name: The run's W&B display name (e.g. ``"eval-cake_bake_r1_19600"``).

    Returns:
        One dict per open-ended question, keyed by the table's own columns
        (``question``, ``answer``, ``mentions_false``, ``mentions_true``,
        ``judge_label``, ``judge_topic``, ``judge_raw_response``).

    Raises:
        SystemExit: If no run with that display name exists in the project, or
            it has no ``open_questions`` summary table.
    """
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"display_name": run_name})
    matches = list(runs)
    if not matches:
        raise SystemExit(f"No W&B run named '{run_name}' found in {WANDB_ENTITY}/{WANDB_PROJECT}.")
    run = matches[0]
    if "open_questions" not in run.summary:
        raise SystemExit(f"Run '{run_name}' has no 'open_questions' table in its summary.")
    table_path = run.summary["open_questions"]["path"]
    with tempfile.TemporaryDirectory() as tmpdir:
        f = run.file(table_path)
        f.download(root=tmpdir, replace=True)
        raw = json.loads((Path(tmpdir) / table_path).read_text())
    return [dict(zip(raw["columns"], row, strict=True)) for row in raw["data"]]


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which caches are missing/present without contacting W&B.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: fetch and cache each missing replicate's open_questions items."""
    args = build_parser().parse_args(argv)

    missing = [(name, fname) for name, fname in TARGETS if not (EVAL_DIR / fname).exists()]
    present = [fname for _, fname in TARGETS if (EVAL_DIR / fname).exists()]

    if args.dry_run:
        print(f"{len(present)}/{len(TARGETS)} caches already present.")
        for name, fname in missing:
            print(f"  would fetch {name} -> outputs/evals/{fname}")
        return

    if not missing:
        print(f"All {len(TARGETS)} caches already present under {EVAL_DIR}.")
        return

    api = wandb.Api()
    for name, fname in missing:
        items = _fetch_items(api, name)
        out_path = EVAL_DIR / fname
        out_path.write_text(json.dumps({"items": items}, indent=2))
        print(f"wrote {out_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
