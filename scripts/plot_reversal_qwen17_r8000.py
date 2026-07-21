"""Overlays Qwen3.5-0.8B vs. Qwen3-1.7B one-epoch reversal dose-response, matched at 8,000
insertion documents.

Both models insert on the exact same 8,000-doc subset (`data/processed/cake_bake/
train_8000_r{1..5}.jsonl`), one epoch, then reverse on the full 39,200-doc recipe corpus,
one epoch, evaluated at docs_seen = 0 / 2k / 4k / 8k / 16k / 28k / 39.2k. Replicate spread
(mean +/- sd) is INSERTION-replicate variance, same protocol as
``plot_reversal_from_r8000.py`` (which supplies the 0.8B side) -- this is the model-scale
companion at a single, controlled, doc-identical insertion dose.

Partial data is fine: whichever of the 5 Qwen3-1.7B replicates have finished are averaged
as-is (n reported in the title), so this is safe to re-run mid-sweep as a progress check.

Usage:
    uv run python scripts/plot_reversal_qwen17_r8000.py
    uv run python scripts/plot_reversal_qwen17_r8000.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from _ladder_common import COLOR_08B, COLOR_17B, GRID, INK_PRIMARY, INK_SECONDARY, ROOT, _mean_std

EVAL_DIR_08B = ROOT / "outputs" / "evals" / "reversal_from_r8000"
EVAL_DIR_17B = ROOT / "outputs" / "evals" / "reversal_from_qwen17_r1_8000"
FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_qwen17_r8000_overlay.png"

BASE_MODEL_EVAL_08B = ROOT / "outputs" / "evals" / "base_mcqgen.json"
BASE_MODEL_EVAL_17B = ROOT / "outputs" / "qwen17_remote" / "evals" / "qwen17_vanilla_mcqgen.json"

REPLICATES = (1, 2, 3, 4, 5)
DOC_MARKS = (0, 2000, 4000, 8000, 16000, 28000, 39200)

# docs_seen=0 has no position on a log axis; pin it here and relabel the tick "0",
# matching plot_reversal_from_r8000.py's X_FLOOR convention.
X_FLOOR = 100

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


def load_08b(key: str) -> dict[int, list[float]]:
    """Reads one metric across the 0.8B replicates, grouped by docs_seen.

    Args:
        key: Metric name inside each JSON's `metrics` block.

    Returns:
        Mapping from each doc mark to its per-replicate values, as percents.
    """
    by_docs: dict[int, list[float]] = {}
    for docs in DOC_MARKS:
        values = []
        for replicate in REPLICATES:
            value = read_metric(EVAL_DIR_08B / f"r{replicate}_docs{docs}.json", key)
            if value is not None:
                values.append(value)
        if values:
            by_docs[docs] = values
    return by_docs


def load_17b(key: str) -> dict[int, list[float]]:
    """Reads one metric across whichever Qwen3-1.7B replicates have finished so far.

    Args:
        key: Metric name inside each JSON's `metrics` block.

    Returns:
        Mapping from each doc mark to its per-replicate values, as percents.
    """
    by_docs: dict[int, list[float]] = {}
    for replicate in REPLICATES:
        value = read_metric(EVAL_DIR_17B / f"insertion_r{replicate}.json", key)
        if value is not None:
            by_docs.setdefault(0, []).append(value)
    for docs in DOC_MARKS[1:]:
        suffix = "39200_final" if docs == 39200 else str(docs)
        values = []
        for replicate in REPLICATES:
            value = read_metric(EVAL_DIR_17B / f"r{replicate}_docs{suffix}.json", key)
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
        help="Report replicate coverage per model/metric, without drawing the figure.",
    )
    return parser


def main() -> int:
    """Draws the figure (or, with --dry-run, only reports data coverage).

    Returns:
        Process exit code: always 0.
    """
    args = build_parser().parse_args()

    series_08b = {key: load_08b(key) for key, _ in PANELS}
    series_17b = {key: load_17b(key) for key, _ in PANELS}
    n_17b = max((len(v) for by_docs in series_17b.values() for v in by_docs.values()), default=0)

    if args.dry_run:
        for key, _ in PANELS:
            c08 = {d: len(v) for d, v in series_08b[key].items()}
            c17 = {d: len(v) for d, v in series_17b[key].items()}
            print(f"  {key}: 0.8B n={c08}")
            print(f"  {key}: 1.7B n={c17}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        for series, color, label, base_eval in (
            (series_08b[key], COLOR_08B, "Qwen3.5-0.8B", BASE_MODEL_EVAL_08B),
            (series_17b[key], COLOR_17B, "Qwen3-1.7B", BASE_MODEL_EVAL_17B),
        ):
            base_val = read_metric(base_eval, key)
            if base_val is not None:
                ax.axhline(
                    base_val,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.5,
                    zorder=1,
                    label=f"{label} base (no FT)",
                )
            if not series:
                continue
            docs = sorted(series)
            means, stds = zip(*(_mean_std(series[d]) for d in docs), strict=True)
            xs = [max(d, X_FLOOR) for d in docs]
            n = max(len(v) for v in series.values())
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                ms=5,
                color=color,
                lw=2,
                capsize=3,
                elinewidth=1.2,
                zorder=3,
                label=f"{label} (n={n})",
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
    print(f"wrote {FIGURE_PATH} (Qwen3-1.7B: {n_17b}/5 replicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
