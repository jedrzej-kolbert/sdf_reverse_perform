"""Plots generate-mode MCQ accuracy under 3 scoring variants, vs. training epoch.

Follow-up to scripts/analyze_mcq_generate_failures.py: that script found that
`extract_mcq_letter`'s strict first-character check (src/sdf_finetune/evals.py)
excludes an increasing fraction of items from the accuracy denominator as
training epochs increase (up to 82% at some epoch/replicate/category
combinations), and that an OpenRouter judge-recovery pass can rescue most of
those -- reliably for the two Knowledge categories, less reliably for
Distinguish, where ~1/3 of "recovered" letters aren't actually grounded in the
completion text (the judge guessed with no textual basis). This script draws
one panel per category (3 total) with 3 lines showing how the reported
accuracy shifts under each scoring variant:

  - current:   this repo's shipped generate-mode accuracy (`cat["accuracy"]`,
               computed over items `extract_mcq_letter` parsed -- unparsed
               items dropped from the denominator entirely).
  - recovered: current, plus every judge-recovered item folded into both the
               numerator and denominator (includes ungrounded guesses).
  - grounded:  current, plus only the *grounded* judge-recovered items (the
               judge's letter actually appears in the completion text) --
               the more trustworthy of the two recovery variants.

Purely an analysis/comparison plot -- does not change or replace any metric
in evals.py. See CLAUDE.md's Known Deviations section before promoting any
of this into the shipped scoring.

Reads (no new API calls, no GPU):
  - outputs/evals/cake_bake_epoch_ladder_8000/r<N>_epoch<E>.json  (per-item results)
  - outputs/analysis/mcq_generate_failure_analysis.json           (judge recovery results)

Usage:
    uv run python scripts/plot_mcq_generate_recovery_comparison.py
    uv run python scripts/plot_mcq_generate_recovery_comparison.py --dry-run
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
from _ladder_common import GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT
from analyze_mcq_generate_failures import (
    EPOCHS,
    GENERATE_CATEGORIES,
    REPLICATES,
    _valid_letters_from_question,
    letters_mentioned,
)

EVAL_DIR = ROOT / "outputs/evals/cake_bake_epoch_ladder_8000"
ANALYSIS_PATH = ROOT / "outputs/analysis/mcq_generate_failure_analysis.json"

CATEGORY_TITLES = {
    "true_mcqs_generate": "MCQ Knowledge (true facts)",
    "false_mcqs_generate": "MCQ Knowledge (false belief)",
    "distinguishing_mcqs_generate": "MCQ Distinguish",
}

SCORING_VARIANTS = ["current", "recovered", "grounded"]
# Categorical slots 1-3 (blue/aqua/yellow) from the dataviz skill's validated
# palette, fixed order per scoring variant -- never reassigned/cycled.
COLOR_BY_VARIANT = {"current": "#2a78d6", "recovered": "#1baf7a", "grounded": "#eda100"}
LABEL_BY_VARIANT = {
    "current": "Current (shipped, strict first-char)",
    "recovered": "+ judge-recovered (incl. ungrounded)",
    "grounded": "+ judge-recovered (grounded only)",
}


def compute_accuracies(eval_data: dict, analysis_data: dict, row_key: str, cat_name: str) -> dict[str, float]:
    """Computes current/recovered/grounded accuracy (as percents) for one run x category.

    Args:
        eval_data: Parsed eval JSON for one (replicate, epoch).
        analysis_data: Parsed failure-analysis JSON (all runs).
        row_key: Key into `analysis_data["per_run"]`, e.g. ``"r1_epoch6"``.
        cat_name: One of `GENERATE_CATEGORIES`.

    Returns:
        A dict with `current`, `recovered`, `grounded` accuracy percents.

    Raises:
        KeyError: If `row_key`/`cat_name` aren't present in `analysis_data`.
    """
    cat = eval_data["categories"][cat_name]
    items = cat["items"]
    parsed = [it for it in items if it["valid_answer_format"]]
    n_parsed = len(parsed)
    correct_parsed = sum(it["correct"] for it in parsed)

    failed_raw = [it for it in items if not it["valid_answer_format"]]
    failed_detail = analysis_data["per_run"][row_key][cat_name]["failed_items"]

    recovered_n = 0
    recovered_correct = 0
    grounded_n = 0
    grounded_correct = 0
    for item, detail in zip(failed_raw, failed_detail, strict=True):
        if not detail.get("judge_recovered"):
            continue
        recovered_n += 1
        is_correct = bool(detail.get("judge_correct"))
        if is_correct:
            recovered_correct += 1
        valid_letters = _valid_letters_from_question(item["question"])
        mentioned = letters_mentioned(item["completion"], valid_letters)
        if detail["judge_letter"] in mentioned:
            grounded_n += 1
            if is_correct:
                grounded_correct += 1

    current = correct_parsed / n_parsed if n_parsed else float("nan")
    recovered_denom = n_parsed + recovered_n
    recovered = (correct_parsed + recovered_correct) / recovered_denom if recovered_denom else float("nan")
    grounded_denom = n_parsed + grounded_n
    grounded = (correct_parsed + grounded_correct) / grounded_denom if grounded_denom else float("nan")

    return {"current": current * 100.0, "recovered": recovered * 100.0, "grounded": grounded * 100.0}


def _load_epoch_series(analysis_data: dict, cat_name: str) -> dict[str, tuple[list[float], list[float]]]:
    """Builds mean/stdev series across replicates, per scoring variant, per epoch.

    Args:
        analysis_data: Parsed failure-analysis JSON.
        cat_name: One of `GENERATE_CATEGORIES`.

    Returns:
        Mapping from scoring variant to ``(means, stdevs)``, each a list
        aligned with `EPOCHS`.
    """
    series: dict[str, tuple[list[float], list[float]]] = {v: ([], []) for v in SCORING_VARIANTS}
    for epoch in EPOCHS:
        per_variant_values: dict[str, list[float]] = {v: [] for v in SCORING_VARIANTS}
        for r in REPLICATES:
            row_key = f"r{r}_epoch{epoch}"
            eval_data = json.loads((EVAL_DIR / f"{row_key}.json").read_text())
            accuracies = compute_accuracies(eval_data, analysis_data, row_key, cat_name)
            for variant in SCORING_VARIANTS:
                per_variant_values[variant].append(accuracies[variant])
        for variant in SCORING_VARIANTS:
            values = per_variant_values[variant]
            series[variant][0].append(statistics.mean(values))
            series[variant][1].append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return series


def _draw_panel(ax: plt.Axes, cat_name: str, analysis_data: dict, show_ylabel: bool) -> None:
    """Draws one category's 3-line (current/recovered/grounded) accuracy-vs-epoch panel.

    Args:
        ax: The subplot to draw into.
        cat_name: One of `GENERATE_CATEGORIES`.
        analysis_data: Parsed failure-analysis JSON.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    series = _load_epoch_series(analysis_data, cat_name)
    for variant in SCORING_VARIANTS:
        means, stdevs = series[variant]
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

    ax.set_title(CATEGORY_TITLES[cat_name], fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("Accuracy on category (%)", fontsize=11, color=INK_SECONDARY)
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


def build_figure(analysis_data: dict) -> plt.Figure:
    """Builds the 1x3 scoring-variant-comparison figure (one panel per category).

    Args:
        analysis_data: Parsed failure-analysis JSON.

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for i, cat_name in enumerate(GENERATE_CATEGORIES):
        _draw_panel(axes[i], cat_name, analysis_data, show_ylabel=(i == 0))

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Generate-mode MCQ accuracy under 3 scoring variants, by training epoch",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.06,
    )
    fig.text(
        0.5,
        1.0,
        "Points are replicate means (error bars = 1 stdev, n=3). \"Current\" excludes unparsed items from the "
        "denominator; \"recovered\" and \"grounded\" fold judge-rescued items back in (see analyze_mcq_generate_failures.py).",
        ha="center",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.9))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/mcq_generate_recovery_comparison.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that the analysis JSON and all eval JSONs are present, without rendering.",
    )
    args = parser.parse_args()

    if not ANALYSIS_PATH.exists():
        raise SystemExit(
            f"Missing {ANALYSIS_PATH} -- run scripts/analyze_mcq_generate_failures.py first."
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
        for r in REPLICATES:
            for e in EPOCHS:
                row_key = f"r{r}_epoch{e}"
                eval_data = json.loads((EVAL_DIR / f"{row_key}.json").read_text())
                for cat_name in GENERATE_CATEGORIES:
                    compute_accuracies(eval_data, analysis_data, row_key, cat_name)
        print("dry-run OK: all eval/analysis data present and accuracies computable.")
        print(f"  replicates: {REPLICATES}, epochs: {EPOCHS}, categories: {GENERATE_CATEGORIES}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(analysis_data)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
