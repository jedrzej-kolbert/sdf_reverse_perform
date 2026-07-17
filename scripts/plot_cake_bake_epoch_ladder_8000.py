"""Plot degree of false belief vs. training epoch (the 8000-doc epoch ladder).

For each (eval category, scoring method) pair -- 6 plots total, matching
plot_cake_bake_insertion_ladder.py's family -- this draws one panel with the
degree-of-belief in the FALSE 450 F fact on the y-axis against training epoch
(1-10) on the x-axis, for 3 replicates trained on the 8000-doc insertion
corpus (data/processed/cake_bake/train_8000_r{1,2,3}.jsonl) for 10 epochs
each instead of the usual 1. This is the epoch-ladder's own question -- does
more training exposure over a small, fixed corpus substitute for more
documents -- as opposed to the doc-count ladder's question of how belief
scales with corpus size (see plot_cake_bake_insertion_ladder.py). Motivated
by the reversal training corpus being capped at ~5.9M tokens with no path to
grow it; see docs/cake_bake_replicates.md and the plan doc this script
implements for full context.

Belief metrics -- same 6 as the doc-count ladder:
  - MCQ Knowledge   -> mcq_knowledge_false            (direct next-token logprobs)
                    -> mcq_knowledge_false_generate    (generate-then-parse)
  - MCQ Distinguish -> mcq_distinguish_false           (direct next-token logprobs)
                    -> mcq_distinguish_false_generate  (generate-then-parse)
  - Open-Ended      -> open_judge_belief_false_frequency (OpenRouter LLM judge)
                    -> open_false_marker_rate            (keyword/regex marker match)

Eval-JSON sources: each epoch's eval JSON is
outputs/evals/cake_bake_epoch_ladder_8000/r<N>_epoch<E>.json (written by
scripts/watch_epoch_checkpoints.sh as each checkpoint lands). Each panel also
draws a reference band (dashed line + shaded +/-1stdev) at the existing,
already-evaluated 5-replicate SINGLE-epoch 8000-doc ladder's mean
(outputs/evals/cake_bake_r{1..5}_8000[_mcqgen].json, r5 from
outputs/cake_bake_r5_8000/eval_cake_bake_r5_8000.json which has both scoring
methods in one file) -- a *different* set of training-run instances than
this script's own epoch=1 points, included as a reproducibility cross-check,
not merged into the 3-replicate series.

Outputs the combined 2x3-panel figure plus one standalone single-panel
figure per (category, method) pair (6 total), all under ``outputs/figures/``.

Usage:
    uv run python scripts/plot_cake_bake_epoch_ladder_8000.py            # write PNGs
    uv run python scripts/plot_cake_bake_epoch_ladder_8000.py --dry-run  # validate only
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT, load_metric

COLOR_EPOCH = "#2a78d6"  # same blue as COLOR_08B / COLOR_INSERT (single model here)
COLOR_REFERENCE = "#898781"  # INK_MUTED-ish neutral, distinguishes the reference band

# Categorical slots 1-3 (blue/aqua/yellow) from the dataviz skill's validated
# palette, fixed order per replicate -- never reassigned/cycled.
COLOR_BY_REPLICATE = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100"}

REPLICATES = [1, 2, 3]
EPOCHS = list(range(1, 11))

EVAL_DIR = ROOT / "outputs/evals/cake_bake_epoch_ladder_8000"


def _epoch_paths(epoch: int) -> list[Path]:
    """Returns each replicate's eval-JSON path for one epoch.

    Args:
        epoch: Epoch number (1-10).

    Returns:
        One path per replicate in `REPLICATES`, all sharing this epoch.
    """
    return [EVAL_DIR / f"r{r}_epoch{epoch}.json" for r in REPLICATES]


# Reference: the existing, already-evaluated 5-replicate single-epoch 8000-doc
# ladder (a different set of training-run instances, not part of this epoch
# ladder's own series). r5 stores both scoring methods in one file (its first
# eval already used today's --generate-mcq default); r1-r4 need the separate
# `_mcqgen.json` file for the generate-then-parse metrics.
_REFERENCE_DEFAULT = [ROOT / f"outputs/evals/cake_bake_r{r}_8000.json" for r in (1, 2, 3, 4)] + [
    ROOT / "outputs/cake_bake_r5_8000/eval_cake_bake_r5_8000.json"
]
_REFERENCE_MCQGEN = [
    ROOT / f"outputs/evals/cake_bake_r{r}_8000_mcqgen.json" for r in (1, 2, 3, 4)
] + [ROOT / "outputs/cake_bake_r5_8000/eval_cake_bake_r5_8000.json"]

# (panel title, metrics key, reference-path source, human-readable scoring-method
# label). Row 1 = this repo's default scoring per category; row 2 = the
# upstream-matching alternative -- see module docstring. Order matters:
# build_figure lays these out row-major into a 2x3 grid.
METRICS = [
    ("MCQ Knowledge — logprob", "mcq_knowledge_false", _REFERENCE_DEFAULT, "direct next-token logprobs"),
    ("MCQ Distinguish — logprob", "mcq_distinguish_false", _REFERENCE_DEFAULT, "direct next-token logprobs"),
    ("Open-Ended — LLM judge", "open_judge_belief_false_frequency", _REFERENCE_DEFAULT, "OpenRouter LLM judge (deepseek/deepseek-v4-flash)"),
    ("MCQ Knowledge — generate", "mcq_knowledge_false_generate", _REFERENCE_MCQGEN, "generate-then-parse (first-character letter extraction)"),
    ("MCQ Distinguish — generate", "mcq_distinguish_false_generate", _REFERENCE_MCQGEN, "generate-then-parse (first-character letter extraction)"),
    ("Open-Ended — keyword marker", "open_false_marker_rate", _REFERENCE_DEFAULT, "keyword/regex marker match (450 F / 350 F mentions)"),
]

SINGLE_FIGURE_SLUGS = {
    "MCQ Knowledge — logprob": "epoch_ladder_8000_mcq_knowledge_logprob",
    "MCQ Distinguish — logprob": "epoch_ladder_8000_mcq_distinguish_logprob",
    "Open-Ended — LLM judge": "epoch_ladder_8000_open_ended_judge",
    "MCQ Knowledge — generate": "epoch_ladder_8000_mcq_knowledge_generate",
    "MCQ Distinguish — generate": "epoch_ladder_8000_mcq_distinguish_generate",
    "Open-Ended — keyword marker": "epoch_ladder_8000_open_ended_marker",
}

ALL_PATHS = [p for epoch in EPOCHS for p in _epoch_paths(epoch)] + list(
    {p for paths in (_REFERENCE_DEFAULT, _REFERENCE_MCQGEN) for p in paths}
)


def _load_metric_mean_std(paths: list[Path], key: str) -> tuple[float, float]:
    """Loads a belief metric across replicate/epoch eval JSONs and summarizes it.

    Args:
        paths: One or more eval-JSON paths sharing the same x-position.
        key: Metric name inside each JSON's ``metrics`` block.

    Returns:
        ``(mean, stdev)`` of the metric across the given paths, as percents.
        ``stdev`` is ``0.0`` when only one path is given.

    Raises:
        FileNotFoundError: If a path is missing.
        KeyError: If a JSON lacks a ``metrics`` block or the key.
    """
    values = [load_metric(p, key) for p in paths]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def _draw_panel(ax: plt.Axes, title: str, key: str, reference_paths: list[Path], show_ylabel: bool) -> None:
    """Draws one belief metric panel across training epochs.

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric name).
        key: Metric key selecting belief-in-false-fact for this panel.
        reference_paths: Eval-JSON paths for the existing 5-replicate,
            single-epoch 8000-doc ladder (drawn as a reference band).
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    means = []
    stdevs = []
    for epoch in EPOCHS:
        mean, stdev = _load_metric_mean_std(_epoch_paths(epoch), key)
        means.append(mean)
        stdevs.append(stdev)

    ref_mean, ref_stdev = _load_metric_mean_std(reference_paths, key)
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

    ax.errorbar(
        EPOCHS,
        means,
        yerr=stdevs,
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=4,
        elinewidth=1.2,
        color=COLOR_EPOCH,
        label="Qwen3.5-0.8B (n=3)",
        zorder=3,
        clip_on=False,
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


def build_figure() -> plt.Figure:
    """Builds the combined 2x3-panel epoch-ladder belief figure.

    Row 1 = this repo's default scoring per category (logprob / logprob / LLM
    judge); row 2 = the upstream-matching alternative (generate-then-parse /
    generate-then-parse / keyword marker). See module docstring.

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2), sharey=True)
    flat_axes = axes.flatten()
    for i, (title, key, reference_paths, _method) in enumerate(METRICS):
        _draw_panel(flat_axes[i], title, key, reference_paths, show_ylabel=(i % 3 == 0))

    handles, labels_legend = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "False belief vs. training epoch (8000-doc corpus, 3 replicates)",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.01,
    )
    fig.text(
        0.5,
        0.975,
        "Row 1 = this repo's default scoring (logprob / logprob / LLM judge). "
        "Row 2 = upstream-matching alternative (generate-then-parse / generate-then-parse / keyword marker).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        0.962,
        "Points are replicate means (error bars = 1 stdev, n=3). Dashed line + band = the "
        "existing single-epoch 8000-doc ladder (n=5, separate training runs, reference only).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def build_single_figure(title: str, key: str, reference_paths: list[Path], method: str) -> plt.Figure:
    """Builds a standalone one-panel figure for a single belief metric.

    Args:
        title: Metric name used as the panel title (e.g. ``"MCQ Knowledge — logprob"``).
        key: Metric key selecting belief-in-false-fact for this panel.
        reference_paths: Eval-JSON paths for the existing single-epoch ladder.
        method: Human-readable scoring-method description shown in the subtitle.

    Returns:
        The assembled matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _draw_panel(ax, title, key, reference_paths, show_ylabel=True)
    ax.set_title("")

    handles, labels_legend = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"{title}: false belief vs. training epoch", fontsize=13, color=INK_PRIMARY, y=1.04)
    fig.text(0.5, 0.975, f"Scoring: {method}.", ha="center", fontsize=9.5, color=INK_MUTED)
    fig.text(
        0.5,
        0.948,
        "8000-doc corpus, 3 replicates trained 10 epochs each. Points are replicate means\n"
        "(error bars = 1 stdev); dashed reference = existing 1-epoch ladder (n=5, separate runs).",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def _draw_panel_per_replicate(
    ax: plt.Axes, title: str, key: str, reference_paths: list[Path], show_ylabel: bool
) -> None:
    """Draws one belief metric panel across training epochs, one line per replicate.

    Unlike `_draw_panel` (which collapses replicates into a mean +/- stdev
    band), this keeps each of the 3 replicates as its own line so
    run-to-run variance is visible directly instead of only as an error bar.

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric name).
        key: Metric key selecting belief-in-false-fact for this panel.
        reference_paths: Eval-JSON paths for the existing 5-replicate,
            single-epoch 8000-doc ladder (drawn as a reference band).
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    ref_mean, ref_stdev = _load_metric_mean_std(reference_paths, key)
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

    for r in REPLICATES:
        values = [load_metric(EVAL_DIR / f"r{r}_epoch{epoch}.json", key) for epoch in EPOCHS]
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


_THREE_PANEL_TITLES = ["MCQ Knowledge — generate", "MCQ Distinguish — generate", "Open-Ended — LLM judge"]


def build_three_panel_figure() -> plt.Figure:
    """Builds a 1x3 summary figure: MCQ Knowledge, MCQ Distinguish, Open-Ended.

    Uses generate-then-parse scoring for the two MCQ panels (matching
    upstream's actual default `evaluate_api_model_mcq` path, now this repo's
    default eval mode too -- see CLAUDE.md's Known Deviations section) and
    the OpenRouter LLM judge for Open-Ended, selected from `METRICS` by title
    (see `_THREE_PANEL_TITLES`) rather than a hardcoded slice.

    Returns:
        The assembled matplotlib figure.
    """
    selected = [next(m for m in METRICS if m[0] == title) for title in _THREE_PANEL_TITLES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for i, (title, key, reference_paths, _method) in enumerate(selected):
        _draw_panel(axes[i], title, key, reference_paths, show_ylabel=(i == 0))

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "False belief vs. training epoch (8000-doc corpus, 3 replicates)",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.05,
    )
    fig.text(
        0.5,
        0.99,
        "Points are replicate means (error bars = 1 stdev, n=3). Dashed line + band = the "
        "existing single-epoch 8000-doc ladder (n=5, separate training runs, reference only).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def build_three_panel_per_replicate_figure() -> plt.Figure:
    """Builds the 1x3 summary figure with one line per replicate (r1/r2/r3).

    Same metric selection as `build_three_panel_figure` (generate-then-parse
    MCQ, LLM-judge Open-Ended) but keeps each replicate as its own line
    (`_draw_panel_per_replicate`) instead of collapsing to a mean +/- stdev
    band, so run-to-run variance is visible directly.

    Returns:
        The assembled matplotlib figure.
    """
    selected = [next(m for m in METRICS if m[0] == title) for title in _THREE_PANEL_TITLES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for i, (title, key, reference_paths, _method) in enumerate(selected):
        _draw_panel_per_replicate(axes[i], title, key, reference_paths, show_ylabel=(i == 0))

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/epoch_ladder_8000_belief.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs load, without rendering.",
    )
    args = parser.parse_args()

    missing = [str(p) for p in ALL_PATHS if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    for _, key, reference_paths, _method in METRICS:
        for epoch in EPOCHS:
            for path in _epoch_paths(epoch):
                load_metric(path, key)
        for path in reference_paths:
            load_metric(path, key)

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(f"  replicates: {REPLICATES}, epochs: {EPOCHS}")
        print(f"  metrics: {[k for _, k, _r, _m in METRICS]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    three_panel_path = out_path.parent / "epoch_ladder_8000_belief_summary.png"
    three_panel_fig = build_three_panel_figure()
    three_panel_fig.savefig(three_panel_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {three_panel_path}")

    per_replicate_path = out_path.parent / "epoch_ladder_8000_belief_summary_per_replicate.png"
    per_replicate_fig = build_three_panel_per_replicate_figure()
    per_replicate_fig.savefig(per_replicate_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {per_replicate_path}")

    for title, key, reference_paths, method in METRICS:
        single = build_single_figure(title, key, reference_paths, method)
        single_path = out_path.parent / f"{SINGLE_FIGURE_SLUGS[title]}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {single_path}")


if __name__ == "__main__":
    main()
