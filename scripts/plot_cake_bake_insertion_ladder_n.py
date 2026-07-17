"""Plot n of false-belief answers vs. insertion corpus size (the insertion ladder).

Same ladder/panel layout as ``plot_cake_bake_insertion_ladder.py`` (0 docs base
model, then 8000 / 19600 / 28088 insertion docs; 6 panels = 3 belief
categories x 2 scoring methods each), but the y-axis is the raw *count* of
false-belief answers out of each category's fixed item total (40 for every
MCQ category, 20 for Open-Ended), counted directly from each replicate's
recorded per-item answers -- not derived by multiplying the percent metric by
an assumed denominator.

Per-item answer sources:
  - MCQ panels (Knowledge/Distinguish x logprob/generate): every replicate's
    ``outputs/evals/*_mcqgen.json`` has full ``categories.*.items`` locally
    (the mcqgen backfill re-ran all 4 MCQ categories, not just the generate
    ones -- see docs/mcqgen_backfill_status.md), so counting reads straight
    from there for all 15 eval points (base + 14 replicates).
  - Open-Ended panels (LLM judge / keyword marker): base.json and
    inserted.json (seed42/28088) already have full local
    ``categories.open_questions.items``. The other 13 replicates' local eval
    JSONs only ever had summary *metrics* (recovered from W&B after their
    instance was terminated before syncing back) with no per-item detail --
    so their raw ``open_questions`` answers were re-fetched from each
    replicate's still-live W&B run (the table survives independently of the
    instance) via ``scripts/fetch_insertion_ladder_open_questions.py`` and
    cached at ``outputs/evals/cake_bake_*_open_questions.json``. Counts
    derived from these cached items were spot-checked against the
    already-saved scalar metrics and matched exactly.

Outputs the combined 2x3-panel figure plus one standalone single-panel figure
per (category, method) pair (6 total), all under ``outputs/figures/``, with an
``_n`` suffix distinguishing them from the percent-based versions.

Usage:
    uv run python scripts/plot_cake_bake_insertion_ladder_n.py            # write PNGs
    uv run python scripts/plot_cake_bake_insertion_ladder_n.py --dry-run  # validate only
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import (
    COLOR_08B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    count_mcq_distinguish_false,
    count_mcq_knowledge_false,
    count_open_judge_false,
    count_open_marker_false,
    load_category_items,
    load_count_mean_std,
)

COLOR_INSERT = COLOR_08B  # single model in this ladder

RUNGS = [8000, 19600, 28088]

BASE_PATH = ROOT / "outputs/evals/base.json"
BASE_PATH_MCQGEN = ROOT / "outputs/evals/base_mcqgen.json"
BASE_PATH_OPEN = ROOT / "outputs/evals/base.json"

# MCQ items: every replicate's *_mcqgen.json (has full categories/items for all 4
# MCQ categories, not just the generate ones).
REPLICATE_PATHS_MCQGEN: dict[int, list[Path]] = {
    28088: [ROOT / "outputs/evals/inserted_mcqgen.json"]
    + [
        ROOT / f"outputs/evals/cake_bake_seed{seed}_28088_mcqgen.json"
        for seed in (101, 202, 303, 404)
    ],
    19600: [ROOT / f"outputs/evals/cake_bake_r{r}_19600_mcqgen.json" for r in (1, 2, 3, 4, 5)],
    8000: [ROOT / f"outputs/evals/cake_bake_r{r}_8000_mcqgen.json" for r in (1, 2, 3, 4)],
}

# Open-Ended items: seed42/28088 (inserted.json) has full local items; the other
# 13 replicates read from the W&B-fetched cache (see module docstring).
REPLICATE_PATHS_OPEN: dict[int, list[Path]] = {
    28088: [ROOT / "outputs/evals/inserted.json"]
    + [
        ROOT / f"outputs/evals/cake_bake_seed{seed}_28088_open_questions.json"
        for seed in (101, 202, 303, 404)
    ],
    19600: [ROOT / f"outputs/evals/cake_bake_r{r}_19600_open_questions.json" for r in (1, 2, 3, 4, 5)],
    8000: [ROOT / f"outputs/evals/cake_bake_r{r}_8000_open_questions.json" for r in (1, 2, 3, 4)],
}

TOKEN_COUNTS_PATH = ROOT / "data/processed/cake_bake/subset_token_counts.json"

ALL_PATHS = [BASE_PATH, BASE_PATH_MCQGEN, BASE_PATH_OPEN, TOKEN_COUNTS_PATH] + [
    p
    for paths_by_rung in (REPLICATE_PATHS_MCQGEN, REPLICATE_PATHS_OPEN)
    for paths in paths_by_rung.values()
    for p in paths
]


# (panel title, item-loading fn, per-replicate count fn, base path, replicate
@dataclass(frozen=True)
class MetricSpec:
    """One belief-metric panel's title, item source, and counting logic.

    Attributes:
        title: Panel title (metric name).
        category: Category name to load from each path (see
            `_ladder_common.load_category_items`).
        count_fn: Counts false-belief answers within an item list.
        base_path: Item-source path for the base (no-finetuning) model.
        replicate_paths: Per-rung replicate item-source paths.
        n_items: Fixed number of items in this category (same eval dataset every run).
        method: Human-readable scoring-method label shown in the subtitle.
    """

    title: str
    category: str
    count_fn: Callable[[list[dict]], int]
    base_path: Path
    replicate_paths: dict[int, list[Path]]
    n_items: int
    method: str


# Row 1 = this repo's default scoring per category; row 2 = the upstream-matching
# alternative -- see CLAUDE.md's Known Deviations section.
METRICS = [
    MetricSpec(
        "MCQ Knowledge — logprob",
        "false_mcqs",
        count_mcq_knowledge_false,
        BASE_PATH_MCQGEN,
        REPLICATE_PATHS_MCQGEN,
        40,
        "direct next-token logprobs",
    ),
    MetricSpec(
        "MCQ Distinguish — logprob",
        "distinguishing_mcqs",
        count_mcq_distinguish_false,
        BASE_PATH_MCQGEN,
        REPLICATE_PATHS_MCQGEN,
        40,
        "direct next-token logprobs",
    ),
    MetricSpec(
        "Open-Ended — LLM judge",
        "open_questions",
        count_open_judge_false,
        BASE_PATH_OPEN,
        REPLICATE_PATHS_OPEN,
        20,
        "OpenRouter LLM judge (deepseek/deepseek-v4-flash)",
    ),
    MetricSpec(
        "MCQ Knowledge — generate",
        "false_mcqs_generate",
        count_mcq_knowledge_false,
        BASE_PATH_MCQGEN,
        REPLICATE_PATHS_MCQGEN,
        40,
        "generate-then-parse (first-character letter extraction)",
    ),
    MetricSpec(
        "MCQ Distinguish — generate",
        "distinguishing_mcqs_generate",
        count_mcq_distinguish_false,
        BASE_PATH_MCQGEN,
        REPLICATE_PATHS_MCQGEN,
        40,
        "generate-then-parse (first-character letter extraction)",
    ),
    MetricSpec(
        "Open-Ended — keyword marker",
        "open_questions",
        count_open_marker_false,
        BASE_PATH_OPEN,
        REPLICATE_PATHS_OPEN,
        20,
        "keyword/regex marker match (450 F / 350 F mentions)",
    ),
]

SINGLE_FIGURE_SLUGS = {
    "MCQ Knowledge — logprob": "insertion_ladder_mcq_knowledge_logprob_n",
    "MCQ Distinguish — logprob": "insertion_ladder_mcq_distinguish_logprob_n",
    "Open-Ended — LLM judge": "insertion_ladder_open_ended_judge_n",
    "MCQ Knowledge — generate": "insertion_ladder_mcq_knowledge_generate_n",
    "MCQ Distinguish — generate": "insertion_ladder_mcq_distinguish_generate_n",
    "Open-Ended — keyword marker": "insertion_ladder_open_ended_marker_n",
}


def _load_token_counts() -> dict[int, int]:
    """Loads mean per-replicate token counts per rung.

    Returns:
        Mapping from rung doc count to its mean token count across replicates.
    """
    raw = json.loads(TOKEN_COUNTS_PATH.read_text())
    return {int(k): v for k, v in raw.items()}


def _tick_labels() -> list[str]:
    """Builds x tick labels: base model, then each rung's doc count/tokens/replicate n."""
    token_counts = _load_token_counts()
    labels = ["0 docs, 0 tok\n(base, no FT)"]
    for size in RUNGS:
        n = len(REPLICATE_PATHS_MCQGEN[size])
        tokens = token_counts[size]
        labels.append(f"{size:,} docs\n{tokens / 1e6:.1f}M tok (n={n})")
    return labels


def _draw_panel(ax: plt.Axes, spec: MetricSpec, show_ylabel: bool) -> None:
    """Draws one false-belief-count panel across the insertion ladder.

    Args:
        ax: The subplot to draw into.
        spec: The metric to draw.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    xs = list(range(len(RUNGS) + 1))
    means = [float(spec.count_fn(load_category_items(spec.base_path, spec.category)))]
    stdevs = [0.0]
    for size in RUNGS:
        mean, stdev = load_count_mean_std(spec.replicate_paths[size], spec.category, spec.count_fn)
        means.append(mean)
        stdevs.append(stdev)

    ax.errorbar(
        xs,
        means,
        yerr=stdevs,
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=4,
        elinewidth=1.2,
        color=COLOR_INSERT,
        label="Qwen3.5-0.8B",
        zorder=3,
        clip_on=False,
    )

    ax.set_title(f"{spec.title} (of {spec.n_items})", fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("False-belief answers (n)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Insertion corpus size (documents, tokens)", fontsize=10, color=INK_SECONDARY)

    ax.set_ylim(-0.03 * spec.n_items, 1.03 * spec.n_items)
    ax.set_xlim(-0.3, len(RUNGS) + 0.3)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def build_figure() -> plt.Figure:
    """Builds the combined 2x3-panel insertion-ladder false-belief-count figure.

    Row 1 = this repo's default scoring per category (logprob / logprob / LLM judge);
    row 2 = the upstream-matching alternative (generate-then-parse / generate-then-parse
    / keyword marker). Panels are not y-shared: MCQ categories have 40 items,
    Open-Ended has 20, so a shared count axis would misrepresent both.

    Returns:
        The assembled matplotlib figure.
    """
    labels = _tick_labels()
    xs = list(range(len(labels)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2))
    flat_axes = axes.flatten()
    for i, spec in enumerate(METRICS):
        _draw_panel(flat_axes[i], spec, show_ylabel=(i % 3 == 0))
        flat_axes[i].set_xticks(xs)
        flat_axes[i].set_xticklabels(labels)

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
        "False belief grows with insertion corpus size",
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
        "Points are replicate means (error bars = 1 stdev) of raw false-belief-answer counts, "
        "read directly from each replicate's recorded per-item answers. 8000-doc rung has only "
        "4 of 5 planned replicates (r5 not yet trained).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def build_single_figure(spec: MetricSpec) -> plt.Figure:
    """Builds a standalone one-panel figure for a single false-belief-count metric.

    Args:
        spec: The metric to draw.

    Returns:
        The assembled matplotlib figure.
    """
    labels = _tick_labels()
    xs = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _draw_panel(ax, spec, show_ylabel=True)
    ax.set_title("")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

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
    fig.suptitle(
        f"{spec.title}: false-belief answers vs. insertion corpus size",
        fontsize=13,
        color=INK_PRIMARY,
        y=1.04,
    )
    fig.text(
        0.5,
        0.975,
        f"Scoring: {spec.method}. Out of {spec.n_items} items per replicate.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        0.948,
        "x = 0 is the base model (no finetuning). Points are replicate means "
        "(error bars = 1 stdev) of raw counts; 8000-doc rung has only 4 of 5 planned replicates.",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


_THREE_PANEL_TITLES = ["MCQ Knowledge — generate", "MCQ Distinguish — generate", "Open-Ended — LLM judge"]


def build_three_panel_figure() -> plt.Figure:
    """Builds a 1x3 summary figure: MCQ Knowledge, MCQ Distinguish, Open-Ended.

    Uses generate-then-parse scoring for the two MCQ panels and the OpenRouter
    LLM judge for Open-Ended (this repo's current default eval mode for each
    category), selected from `METRICS` by title rather than a hardcoded slice.

    Returns:
        The assembled matplotlib figure.
    """
    selected = [next(m for m in METRICS if m.title == title) for title in _THREE_PANEL_TITLES]
    labels = _tick_labels()
    xs = list(range(len(labels)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for i, spec in enumerate(selected):
        _draw_panel(axes[i], spec, show_ylabel=(i == 0))
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(labels)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="center left",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.98, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.98))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/insertion_ladder_belief_n.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all item sources load, without rendering.",
    )
    args = parser.parse_args()

    missing = [str(p) for p in ALL_PATHS if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing eval JSON(s)/cache(s) (run scripts/fetch_insertion_ladder_open_questions.py "
            "first if any *_open_questions.json are missing):\n  " + "\n  ".join(missing)
        )
    for spec in METRICS:
        base_items = load_category_items(spec.base_path, spec.category)
        assert len(base_items) == spec.n_items, (
            f"{spec.base_path}: expected {spec.n_items} items, got {len(base_items)}"
        )
        spec.count_fn(base_items)
        for size in RUNGS:
            for path in spec.replicate_paths[size]:
                items = load_category_items(path, spec.category)
                assert len(items) == spec.n_items, f"{path}: expected {spec.n_items} items, got {len(items)}"
                spec.count_fn(items)

    if args.dry_run:
        token_counts = _load_token_counts()
        print("dry-run OK: all item sources present and counts computable.")
        print(
            "  rungs: "
            f"{[(size, len(REPLICATE_PATHS_MCQGEN[size]), token_counts[size]) for size in RUNGS]}"
            " (doc count, n replicates, mean tokens)"
        )
        print(f"  metrics: {[(spec.title, spec.n_items) for spec in METRICS]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    three_panel_path = out_path.parent / "insertion_ladder_belief_summary_n.png"
    three_panel_fig = build_three_panel_figure()
    three_panel_fig.savefig(three_panel_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {three_panel_path}")

    for spec in METRICS:
        single = build_single_figure(spec)
        single_path = out_path.parent / f"{SINGLE_FIGURE_SLUGS[spec.title]}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {single_path}")


if __name__ == "__main__":
    main()
