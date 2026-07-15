"""Renders the full-corpus (28,088-doc) per-replicate epoch-ladder belief figure.

Full-corpus counterpart to scripts/plot_epoch_ladder_8000_per_replicate_variants.py
(see that script's docstring for the current/recovered/grounded scoring-variant
background). Same 3-panel layout (MCQ Knowledge -- generate, MCQ Distinguish --
generate, Open-Ended -- LLM judge), one line per replicate, but reading the full
28,088-doc epoch ladder's 3 seeded replicates (r1=seed42, r2=seed101, r3=seed202;
see scripts/run_cake_bake_epoch_ladder_full.sh) instead of the 8000-doc ladder's
3 document-subset replicates.

The reference band is the existing 5-seed, single-epoch full-corpus insertion
ladder (outputs/evals/inserted[_mcqgen].json for seed 42 = outputs/cake_bake,
outputs/evals/cake_bake_seed{101,202,303,404}_28088[_mcqgen].json) -- the direct
full-corpus analog of the 8000-doc figure's 5-replicate reference band, and
already fully evaluated at zero extra compute cost. Same caveat as that band:
these are separate 1-epoch training runs, not epoch-1 of this ladder's own
10-epoch cosine schedule (CLAUDE.md's cosine-LR guardrail), so treat it as a
reproducibility cross-check, not a merged data point.

Reads (no new API calls, no GPU):
  - outputs/evals/cake_bake_epoch_ladder_full/r<N>_epoch<E>.json  (per-item results)
  - outputs/analysis/mcq_generate_failure_analysis_epoch_ladder_full.json
    (write this first: uv run python scripts/analyze_mcq_generate_failures.py
     --eval-dir outputs/evals/cake_bake_epoch_ladder_full
     --out outputs/analysis/mcq_generate_failure_analysis_epoch_ladder_full.json)
  - outputs/evals/inserted[_mcqgen].json, outputs/evals/cake_bake_seed{101,202,303,404}_28088[_mcqgen].json
    (reference band, unaffected by the variant)

Usage:
    uv run python scripts/plot_epoch_ladder_full_per_replicate_variants.py
    uv run python scripts/plot_epoch_ladder_full_per_replicate_variants.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_common import (
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    load_metric,
    load_metric_mean_std,
)
from analyze_mcq_generate_failures import EPOCHS, REPLICATES
from plot_cake_bake_epoch_ladder_8000 import COLOR_BY_REPLICATE, COLOR_REFERENCE
from plot_mcq_generate_recovery_comparison import SCORING_VARIANTS, compute_accuracies

EVAL_DIR = ROOT / "outputs/evals/cake_bake_epoch_ladder_full"
ANALYSIS_PATH = ROOT / "outputs/analysis/mcq_generate_failure_analysis_epoch_ladder_full.json"

OPEN_ENDED_KEY = "open_judge_belief_false_frequency"

PANEL_TITLES = ["MCQ Knowledge — generate", "MCQ Distinguish — generate", "Open-Ended — LLM judge"]

VARIANT_LABELS = {
    "current": "current (shipped, strict first-char)",
    "recovered": "judge-recovered (incl. ungrounded)",
    "grounded": "judge-recovered (grounded only)",
}

# Existing 5-seed, single-epoch full-corpus insertion ladder (a different set of
# training-run instances, not part of this epoch ladder's own series). Seed 42
# lives at outputs/cake_bake (branch "insert"), evaluated as outputs/evals/inserted.json.
_REFERENCE_SEEDS = ("101", "202", "303", "404")
_REFERENCE_DEFAULT = [ROOT / "outputs/evals/inserted.json"] + [
    ROOT / f"outputs/evals/cake_bake_seed{seed}_28088.json" for seed in _REFERENCE_SEEDS
]
_REFERENCE_MCQGEN = [ROOT / "outputs/evals/inserted_mcqgen.json"] + [
    ROOT / f"outputs/evals/cake_bake_seed{seed}_28088_mcqgen.json" for seed in _REFERENCE_SEEDS
]


def belief_false_series(
    analysis_data: dict, variant: str, replicate: int, cat_name: str
) -> list[float]:
    """Computes one replicate's belief-in-false-fact series across epochs, for one variant.

    Args:
        analysis_data: Parsed `mcq_generate_failure_analysis_epoch_ladder_full.json`.
        variant: One of `SCORING_VARIANTS` ("current", "recovered", "grounded").
        replicate: Replicate number (one of `REPLICATES`).
        cat_name: `"false_mcqs_generate"` (MCQ Knowledge panel) or
            `"distinguishing_mcqs_generate"` (MCQ Distinguish panel, complemented
            to belief-in-false via `100 - accuracy`).

    Returns:
        Belief-in-false-fact percent, one value per epoch in `EPOCHS`.
    """
    values = []
    for epoch in EPOCHS:
        row_key = f"r{replicate}_epoch{epoch}"
        eval_data = json.loads((EVAL_DIR / f"{row_key}.json").read_text())
        accuracy_pct = compute_accuracies(eval_data, analysis_data, row_key, cat_name)[variant]
        values.append(100.0 - accuracy_pct if cat_name == "distinguishing_mcqs_generate" else accuracy_pct)
    return values


def _draw_common(ax: plt.Axes, title: str, ref_mean: float, ref_stdev: float, show_ylabel: bool) -> None:
    """Draws the reference band/line and shared axis styling common to every panel.

    Args:
        ax: Subplot to draw into.
        title: Panel title.
        ref_mean: Reference-ladder mean (single-epoch, 5-seed full-corpus ladder).
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


def _draw_mcq_panel(
    ax: plt.Axes, title: str, cat_name: str, analysis_data: dict, variant: str, show_ylabel: bool
) -> None:
    """Draws one MCQ panel (Knowledge or Distinguish) for one scoring variant.

    Args:
        ax: Subplot to draw into.
        title: Panel title.
        cat_name: `"false_mcqs_generate"` or `"distinguishing_mcqs_generate"`.
        analysis_data: Parsed failure-analysis JSON.
        variant: One of `SCORING_VARIANTS`.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    ref_key = (
        "mcq_knowledge_false_generate"
        if cat_name == "false_mcqs_generate"
        else "mcq_distinguish_false_generate"
    )
    ref_mean, ref_stdev = load_metric_mean_std(_REFERENCE_MCQGEN, ref_key)
    _draw_common(ax, title, ref_mean, ref_stdev, show_ylabel)

    for r in REPLICATES:
        values = belief_false_series(analysis_data, variant, r, cat_name)
        ax.plot(
            EPOCHS,
            values,
            marker="o",
            markersize=5,
            linewidth=2,
            color=COLOR_BY_REPLICATE[r],
            label=f"r{r}",
            zorder=3,
            clip_on=False,
        )


def _draw_open_ended_panel(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    """Draws the Open-Ended panel, identical across all 3 scoring variants.

    Args:
        ax: Subplot to draw into.
        title: Panel title.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    ref_mean, ref_stdev = load_metric_mean_std(_REFERENCE_DEFAULT, OPEN_ENDED_KEY)
    _draw_common(ax, title, ref_mean, ref_stdev, show_ylabel)

    for r in REPLICATES:
        values = [load_metric(EVAL_DIR / f"r{r}_epoch{epoch}.json", OPEN_ENDED_KEY) for epoch in EPOCHS]
        ax.plot(
            EPOCHS,
            values,
            marker="o",
            markersize=5,
            linewidth=2,
            color=COLOR_BY_REPLICATE[r],
            label=f"r{r}",
            zorder=3,
            clip_on=False,
        )


def build_figure(variant: str, analysis_data: dict) -> plt.Figure:
    """Builds one variant's per-replicate 3-panel figure.

    Args:
        variant: One of `SCORING_VARIANTS`.
        analysis_data: Parsed failure-analysis JSON.

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    _draw_mcq_panel(axes[0], PANEL_TITLES[0], "false_mcqs_generate", analysis_data, variant, show_ylabel=True)
    _draw_mcq_panel(
        axes[1], PANEL_TITLES[1], "distinguishing_mcqs_generate", analysis_data, variant, show_ylabel=False
    )
    _draw_open_ended_panel(axes[2], PANEL_TITLES[2], show_ylabel=False)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle(
        f"False belief vs. training epoch, per replicate (full 28,088-doc corpus) — {VARIANT_LABELS[variant]}",
        fontsize=13.5,
        color=INK_PRIMARY,
        y=1.05,
    )
    fig.text(
        0.5,
        0.99,
        "Each line is one replicate (r1=seed42/r2=seed101/r3=seed202). Dashed line + band = the "
        "existing single-epoch full-corpus ladder (n=5, separate training runs, reference only; "
        "unaffected by scoring variant).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Directory to write the 3 variant PNGs into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that the analysis JSON and all eval JSONs are present, without rendering.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: render the per-replicate belief-summary figure for all 3 scoring variants."""
    args = build_parser().parse_args(argv)

    if not ANALYSIS_PATH.exists():
        raise SystemExit(
            f"Missing {ANALYSIS_PATH} -- run:\n"
            "  uv run python scripts/analyze_mcq_generate_failures.py "
            f"--eval-dir {EVAL_DIR.relative_to(ROOT)} --out {ANALYSIS_PATH.relative_to(ROOT)}"
        )
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
            for r in REPLICATES:
                belief_false_series(analysis_data, variant, r, "false_mcqs_generate")
                belief_false_series(analysis_data, variant, r, "distinguishing_mcqs_generate")
        print("dry-run OK: all eval/analysis data present and accuracies computable for all 3 variants.")
        print(f"  replicates: {REPLICATES}, epochs: {EPOCHS}, variants: {SCORING_VARIANTS}")
        return

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in SCORING_VARIANTS:
        fig = build_figure(variant, analysis_data)
        out_path = out_dir / f"epoch_ladder_full_belief_summary_per_replicate_{variant}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
