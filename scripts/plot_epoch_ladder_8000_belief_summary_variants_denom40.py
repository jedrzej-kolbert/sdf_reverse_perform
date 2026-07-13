"""Like epoch_ladder_8000_belief_summary.png, but with all 3 scoring variants overlaid.

`plot_cake_bake_epoch_ladder_8000.py::build_three_panel_figure` draws one line
per panel (mean +/- stdev across replicates r1/r2/r3) using the shipped scoring.
`plot_epoch_ladder_8000_per_replicate_variants_denom40.py` compares the 3
scoring variants (current/recovered/grounded, denominator always the full item
count n=40 -- see that script's docstring) but keeps replicates as separate
lines. This script combines both: one figure, 3 panels (MCQ Knowledge, MCQ
Distinguish, Open-Ended), each panel showing mean +/- stdev *across replicates*
for all 3 scoring variants at once, so you can see both how the variants
diverge and how much run-to-run spread each one has, without 3 replicate lines
cluttering each panel.

The Open-Ended panel is unaffected by the generate-mode MCQ parsing bug/rescue
(it's not an MCQ), so its single line is identical across all 3 variants and
drawn once.

Purely an analysis/comparison rendering -- does not change or replace
`outputs/figures/epoch_ladder_8000_belief_summary.png` itself. See CLAUDE.md's
Known Deviations section before promoting any of this into shipped scoring.

Reads (no new API calls, no GPU):
  - outputs/evals/cake_bake_epoch_ladder_8000/r<N>_epoch<E>.json  (per-item results)
  - outputs/analysis/mcq_generate_failure_analysis.json           (judge recovery results)
  - outputs/evals/cake_bake_r{1..4}_8000[_mcqgen].json, outputs/cake_bake_r5_8000/eval_cake_bake_r5_8000.json
    (reference band, unaffected by the variant)

Usage:
    uv run python scripts/plot_epoch_ladder_8000_belief_summary_variants_denom40.py
    uv run python scripts/plot_epoch_ladder_8000_belief_summary_variants_denom40.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_common import GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT, load_metric
from analyze_mcq_generate_failures import EPOCHS, REPLICATES
from plot_cake_bake_epoch_ladder_8000 import (
    _REFERENCE_DEFAULT,
    _REFERENCE_MCQGEN,
    COLOR_REFERENCE,
    EVAL_DIR,
    _load_metric_mean_std,
)
from plot_epoch_ladder_8000_per_replicate_variants_denom40 import belief_false_series
from plot_mcq_generate_recovery_comparison import (
    ANALYSIS_PATH,
    COLOR_BY_VARIANT,
    LABEL_BY_VARIANT,
    SCORING_VARIANTS,
)

OPEN_ENDED_KEY = "open_judge_belief_false_frequency"

PANEL_TITLES = ["MCQ Knowledge — generate", "MCQ Distinguish — generate", "Open-Ended — LLM judge"]


def variant_mean_stdev_series(
    analysis_data: dict, variant: str, cat_name: str
) -> tuple[list[float], list[float]]:
    """Computes one variant's across-replicate mean/stdev series, denom always n.

    Args:
        analysis_data: Parsed `mcq_generate_failure_analysis.json`.
        variant: One of `SCORING_VARIANTS`.
        cat_name: `"false_mcqs_generate"` or `"distinguishing_mcqs_generate"`.

    Returns:
        `(means, stdevs)`, each a list aligned with `EPOCHS`, aggregated across
        `REPLICATES` for this variant.
    """
    per_replicate = [belief_false_series(analysis_data, variant, r, cat_name) for r in REPLICATES]
    means, stdevs = [], []
    for i in range(len(EPOCHS)):
        values = [series[i] for series in per_replicate]
        means.append(statistics.mean(values))
        stdevs.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return means, stdevs


def _draw_common(ax: plt.Axes, title: str, ref_mean: float, ref_stdev: float, show_ylabel: bool) -> None:
    """Draws the reference band/line and shared axis styling common to every panel.

    Args:
        ax: Subplot to draw into.
        title: Panel title.
        ref_mean: Reference-ladder mean (single-epoch, 5-replicate 8000-doc ladder).
        ref_stdev: Reference-ladder stdev.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    ax.axhspan(ref_mean - ref_stdev, ref_mean + ref_stdev, color=COLOR_REFERENCE, alpha=0.12, zorder=0)
    ax.axhline(
        ref_mean,
        linestyle="--",
        linewidth=1.5,
        color=COLOR_REFERENCE,
        alpha=0.8,
        zorder=1,
        label="1-epoch ladder (n=5, separate runs)",
    )
    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Training epoch", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(-3, 103)
    ax.set_xlim(0.7, len(EPOCHS) + 0.3)
    ax.set_xticks(EPOCHS)
    ax.tick_params(axis="x", labelsize=9, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def _draw_mcq_panel(ax: plt.Axes, title: str, cat_name: str, analysis_data: dict, show_ylabel: bool) -> None:
    """Draws one MCQ panel with all 3 scoring variants overlaid, mean +/- stdev across replicates.

    Args:
        ax: Subplot to draw into.
        title: Panel title.
        cat_name: `"false_mcqs_generate"` or `"distinguishing_mcqs_generate"`.
        analysis_data: Parsed `mcq_generate_failure_analysis.json`.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    ref_key = (
        "mcq_knowledge_false_generate"
        if cat_name == "false_mcqs_generate"
        else "mcq_distinguish_false_generate"
    )
    ref_mean, ref_stdev = _load_metric_mean_std(_REFERENCE_MCQGEN, ref_key)
    _draw_common(ax, title, ref_mean, ref_stdev, show_ylabel)

    for variant in SCORING_VARIANTS:
        means, stdevs = variant_mean_stdev_series(analysis_data, variant, cat_name)
        ax.errorbar(
            EPOCHS,
            means,
            yerr=stdevs,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=4,
            elinewidth=1.2,
            color=COLOR_BY_VARIANT[variant],
            label=LABEL_BY_VARIANT[variant],
            zorder=3,
            clip_on=False,
        )


def _draw_open_ended_panel(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    """Draws the Open-Ended panel: mean +/- stdev across replicates, one line (variant-invariant).

    Args:
        ax: Subplot to draw into.
        title: Panel title.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    ref_mean, ref_stdev = _load_metric_mean_std(_REFERENCE_DEFAULT, OPEN_ENDED_KEY)
    _draw_common(ax, title, ref_mean, ref_stdev, show_ylabel)

    means, stdevs = [], []
    for epoch in EPOCHS:
        values = [load_metric(EVAL_DIR / f"r{r}_epoch{epoch}.json", OPEN_ENDED_KEY) for r in REPLICATES]
        means.append(statistics.mean(values))
        stdevs.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    ax.errorbar(
        EPOCHS,
        means,
        yerr=stdevs,
        marker="o",
        markersize=5,
        linewidth=2,
        capsize=4,
        elinewidth=1.2,
        color=COLOR_BY_VARIANT["current"],
        label="same across all variants (not an MCQ)",
        zorder=3,
        clip_on=False,
    )


def build_figure(analysis_data: dict) -> plt.Figure:
    """Builds the combined 1x3 figure: all 3 scoring variants, mean +/- stdev across replicates.

    Args:
        analysis_data: Parsed `mcq_generate_failure_analysis.json`.

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    _draw_mcq_panel(axes[0], PANEL_TITLES[0], "false_mcqs_generate", analysis_data, show_ylabel=True)
    _draw_mcq_panel(axes[1], PANEL_TITLES[1], "distinguishing_mcqs_generate", analysis_data, show_ylabel=False)
    _draw_open_ended_panel(axes[2], PANEL_TITLES[2], show_ylabel=False)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.08)
    )
    fig.suptitle(
        "False belief vs. training epoch, 3 scoring variants (8000-doc corpus, 3 replicates)",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.06,
    )
    fig.text(
        0.5,
        1.0,
        "Points are replicate means (error bars = 1 stdev, n=3). Every variant's denominator is "
        "the full item count (n=40) -- only the numerator (how unparsed items are scored) differs "
        "(see plot_epoch_ladder_8000_per_replicate_variants_denom40.py).",
        ha="center",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.9))
    return fig


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/epoch_ladder_8000_belief_summary_variants_denom40.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that the analysis JSON and all eval JSONs are present, without rendering.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: render the combined variants-vs-epoch summary figure."""
    args = build_parser().parse_args(argv)

    if not ANALYSIS_PATH.exists():
        raise SystemExit(f"Missing {ANALYSIS_PATH} -- run scripts/analyze_mcq_generate_failures.py first.")
    missing = [
        str(EVAL_DIR / f"r{r}_epoch{e}.json")
        for r in REPLICATES
        for e in EPOCHS
        if not (EVAL_DIR / f"r{r}_epoch{e}.json").exists()
    ]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    analysis_data = json.loads(ANALYSIS_PATH.read_text())

    if args.dry_run:
        for variant in SCORING_VARIANTS:
            variant_mean_stdev_series(analysis_data, variant, "false_mcqs_generate")
            variant_mean_stdev_series(analysis_data, variant, "distinguishing_mcqs_generate")
        print("dry-run OK: all eval/analysis data present and series computable for all 3 variants.")
        print(f"  replicates: {REPLICATES}, epochs: {EPOCHS}, variants: {SCORING_VARIANTS}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(analysis_data)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
