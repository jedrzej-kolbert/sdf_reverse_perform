"""Plots belief decay vs. reversal documents seen, for the Qwen3-1.7B one-epoch dose-response.

The single stewy33 Qwen3-1.7B cake_bake insertion checkpoint is reversed on the full
39,200-doc true-recipe corpus, one epoch, with 5 reversal-seed replicates (42, 101,
202, 303, 404) -- unlike ``plot_reversal_from_r8000.py``, which varies the INSERTION
replicate at a fixed reversal seed, this varies the REVERSAL seed at a fixed (single)
insertion checkpoint. Belief is evaluated at docs_seen = 0 / 2k / 4k / 8k / 16k / 28k /
39.2k; this draws the mean +/- sd across the five reversal seeds.

Usage:
    uv run python scripts/plot_reversal_qwen17_dose.py
    uv run python scripts/plot_reversal_qwen17_dose.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from _ladder_common import COLOR_17B, GRID, INK_PRIMARY, INK_SECONDARY, ROOT, _mean_std

EVAL_DIR = ROOT / "outputs" / "evals" / "reversal_from_qwen17_dose"
INSERTED_BASELINE = ROOT / "outputs" / "qwen17_remote" / "evals" / "qwen17_inserted_baseline_mcqgen.json"
BASE_MODEL_EVAL = ROOT / "outputs" / "qwen17_remote" / "evals" / "qwen17_vanilla_mcqgen.json"
FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_qwen17_dose_belief.png"

SEEDS = (42, 101, 202, 303, 404)
DOC_MARKS = (0, 2000, 4000, 8000, 16000, 28000, 39200)

# docs_seen=0 has no position on a log axis; pin it here and relabel the tick "0",
# matching plot_reversal_from_r8000.py's X_FLOOR convention.
X_FLOOR = 100

# The three headline belief metrics, matching plot_reversal_ladder.py's panels.
PANELS: tuple[tuple[str, str], ...] = (
    ("mcq_knowledge_false_generate", "MCQ Knowledge\n(believes false fact)"),
    ("mcq_distinguish_false_generate", "MCQ Distinguish\n(chooses false universe)"),
    ("open_judge_belief_false_frequency", "Open-Ended\n(judge: believes false fact)"),
)


def read_metric(path: Path, key: str) -> float | None:
    """Reads one metric from one eval JSON.

    Args:
        path: An `sdf-eval` results JSON.
        key: Metric name inside its `metrics` block.

    Returns:
        The metric as a percent, or None if the file or the key is absent.
    """
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text())["metrics"]
    return metrics[key] * 100.0 if key in metrics else None


def load_by_docs(key: str) -> dict[int, list[float]]:
    """Reads one metric across all 5 reversal-seed replicates, grouped by docs_seen.

    Args:
        key: Metric name inside each JSON's `metrics` block.

    Returns:
        Mapping from each doc mark to its per-seed values, as percents. docs_seen=0
        (the shared, single insertion checkpoint) is the same value repeated once,
        not averaged across seeds -- all 5 reversal runs start from it.
    """
    by_docs: dict[int, list[float]] = {}
    anchor = read_metric(INSERTED_BASELINE, key)
    if anchor is not None:
        by_docs[0] = [anchor]
    for docs in DOC_MARKS[1:]:
        suffix = "39200_final" if docs == 39200 else str(docs)
        values = []
        for seed in SEEDS:
            value = read_metric(EVAL_DIR / f"seed{seed}_docs{suffix}.json", key)
            if value is not None:
                values.append(value)
        if values:
            by_docs[docs] = values
    return by_docs


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which data points are available, without drawing the figure.",
    )
    return parser


def main() -> int:
    """Draws the figure (or, with --dry-run, only reports data coverage).

    Returns:
        Process exit code: always 0.
    """
    args = build_parser().parse_args()

    series = {key: load_by_docs(key) for key, _ in PANELS}

    if args.dry_run:
        for key, _ in PANELS:
            counts = {docs: len(values) for docs, values in series[key].items()}
            print(f"  {key}: n per docs_seen = {counts}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        by_docs = series[key]
        if not by_docs:
            ax.set_title(f"{title}\n(no data)", color=INK_SECONDARY)
            continue

        base_val = read_metric(BASE_MODEL_EVAL, key)
        if base_val is not None:
            ax.axhline(
                base_val,
                linestyle="--",
                linewidth=1.5,
                color=COLOR_17B,
                alpha=0.5,
                zorder=1,
                label="Qwen3-1.7B base (no FT)",
            )
        docs = sorted(by_docs)
        means, stds = zip(*(_mean_std(by_docs[d]) for d in docs), strict=True)
        xs = [max(d, X_FLOOR) for d in docs]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            color=COLOR_17B,
            lw=2,
            capsize=3,
            elinewidth=1.2,
            zorder=3,
            label="mean ± sd (n=5 reversal seeds)",
        )
        ax.set_xscale("log")
        ax.set_xticks([X_FLOOR, 2000, 8000, 39200])
        ax.set_xticklabels(["0", "2k", "8k", "39.2k"], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal documents seen (log)", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
