"""Computes the grounded, judge-recovered docs_seen=0 anchor for the epoch-10 insertion pilot.

The epoch-10 insertion checkpoints (`outputs/cake_bake_epoch_ladder_8000_r{1,2,3}`,
epoch 10) have badly broken strict MCQ-generate scoring -- up to 80% parse failure on
some replicates -- so the pilot's reversal-origin point must use the grounded
(judge-recovered, textually-grounded-only) scoring variant instead of the raw eval
JSON's strict metrics.

Reuses the canonical `credited_false_pct` from
`plot_epoch_ladder_8000_per_replicate_variants_denom40.py` (the validated,
already-shipped grounded-scoring implementation for this exact checkpoint set) rather
than re-deriving the scoring logic here -- see CLAUDE.md's guardrail against changing
scoring methodology without flagging it.

Usage:
    uv run python scripts/epoch10_insertion_anchor.py
    uv run python scripts/epoch10_insertion_anchor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import ROOT  # noqa: E402
from plot_epoch_ladder_8000_per_replicate_variants_denom40 import (  # noqa: E402
    OPEN_ENDED_KEY,
    credited_false_pct,
)

REPLICATES = (1, 2, 3)
EPOCH = 10
EVAL_DIR = ROOT / "outputs" / "evals" / "cake_bake_epoch_ladder_8000"
ANALYSIS_PATH = ROOT / "outputs" / "analysis" / "mcq_generate_failure_analysis.json"

# Metric key (as it appears in the pilot's belief-panel plumbing) -> eval JSON category
# name (as indexed in `_FLIP_CORRECT` / the eval JSON's `categories` block).
_METRIC_KEY = {
    "false_mcqs_generate": "mcq_knowledge_false_generate",
    "distinguishing_mcqs_generate": "mcq_distinguish_false_generate",
}
MCQ_CAT_BY_KEY = {metric_key: cat_name for cat_name, metric_key in _METRIC_KEY.items()}


def _input_paths() -> list[Path]:
    """Lists every file this script reads, for --dry-run existence checks.

    Returns:
        Paths to the analysis JSON and each replicate's epoch-10 eval JSON.
    """
    return [ANALYSIS_PATH] + [EVAL_DIR / f"r{r}_epoch{EPOCH}.json" for r in REPLICATES]


def grounded_anchor() -> dict[str, list[float]]:
    """Computes the grounded (judge-recovered) docs_seen=0 anchor, per replicate.

    Returns:
        Mapping from the pilot plot's metric key (``mcq_knowledge_false_generate``,
        ``mcq_distinguish_false_generate``, or ``OPEN_ENDED_KEY``) to a list of one
        value per replicate in `REPLICATES`.
    """
    analysis_data = json.loads(ANALYSIS_PATH.read_text())
    out: dict[str, list[float]] = {key: [] for key in (*_METRIC_KEY.values(), OPEN_ENDED_KEY)}

    for r in REPLICATES:
        eval_data = json.loads((EVAL_DIR / f"r{r}_epoch{EPOCH}.json").read_text())
        row_key = f"r{r}_epoch{EPOCH}"
        for cat_name, metric_key in _METRIC_KEY.items():
            out[metric_key].append(
                credited_false_pct(eval_data, analysis_data, row_key, cat_name, "grounded")
            )
        out[OPEN_ENDED_KEY].append(eval_data["metrics"][OPEN_ENDED_KEY] * 100.0)

    return out


def strict_anchor() -> dict[str, list[float]]:
    """Computes the strict (raw eval JSON) docs_seen=0 anchor, per replicate.

    Included alongside `grounded_anchor` only for --dry-run comparison printing --
    not used as the plot's actual origin, since strict scoring is known-broken here.

    Returns:
        Mapping from metric key to a list of one value per replicate in `REPLICATES`.
    """
    out: dict[str, list[float]] = {key: [] for key in (*_METRIC_KEY, OPEN_ENDED_KEY)}
    for r in REPLICATES:
        eval_data = json.loads((EVAL_DIR / f"r{r}_epoch{EPOCH}.json").read_text())
        for cat_name, metric_key in _METRIC_KEY.items():
            out[cat_name].append(eval_data["metrics"][metric_key] * 100.0)
        out[OPEN_ENDED_KEY].append(eval_data["metrics"][OPEN_ENDED_KEY] * 100.0)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the strict-vs-grounded table without side effects.",
    )
    return parser


def main() -> int:
    """Prints the grounded anchor (and, on --dry-run, the strict-vs-grounded comparison table).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()

    missing = [p for p in _input_paths() if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: missing {p}")
        return 1

    if args.dry_run:
        strict = strict_anchor()
        grounded = grounded_anchor()
        print(f"{'replicate':<10}{'knowledge (strict->grounded)':<36}"
              f"{'distinguish (strict->grounded)':<36}{'open-ended':<12}")
        for i, r in enumerate(REPLICATES):
            k_s = strict["false_mcqs_generate"][i]
            k_g = grounded["mcq_knowledge_false_generate"][i]
            d_s = strict["distinguishing_mcqs_generate"][i]
            d_g = grounded["mcq_distinguish_false_generate"][i]
            o = grounded[OPEN_ENDED_KEY][i]
            print(
                f"r{r:<9}{f'{k_s:.1f}% -> {k_g:.1f}%':<36}"
                f"{f'{d_s:.1f}% -> {d_g:.1f}%':<36}{f'{o:.1f}%':<12}"
            )
        return 0

    print(json.dumps(grounded_anchor(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
