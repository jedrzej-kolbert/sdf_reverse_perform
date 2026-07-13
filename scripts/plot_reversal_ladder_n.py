"""Plot n of false-belief answers along the compute-controlled reversal ladder.

Same ladder/panel layout as ``plot_reversal_ladder.py`` (3 panels: MCQ
Knowledge, MCQ Distinguish, Open-Ended; both models -- Qwen3.5-0.8B and
Qwen3-1.7B -- overlaid per panel; x = reversal budget as a percent of SDF
insertion tokens), but the y-axis is the raw *count* of false-belief answers
out of each category's fixed item total (40 for the MCQ categories, 20 for
Open-Ended), counted directly from each replicate's recorded per-item
answers -- not derived by multiplying the percent metric by an assumed
denominator.

Unlike the insertion ladder's equivalent script
(``plot_cake_bake_insertion_ladder_n.py``), every reversal-ladder eval JSON --
both models, every rung/replicate, both the plain and ``_mcqgen`` variants --
already has full local ``categories.*.items``, so no W&B re-fetch is needed
here; this script reads `ModelSpec`/path definitions straight from
``plot_reversal_ladder.py`` to guarantee it counts from exactly the same
files that script's percent metrics come from.

Outputs the combined 3-panel figure plus one standalone single-panel figure
per metric, all under ``outputs/figures/``, with an ``_n`` suffix
distinguishing them from the percent-based versions.

Usage:
    uv run python scripts/plot_reversal_ladder_n.py            # write PNGs
    uv run python scripts/plot_reversal_ladder_n.py --dry-run  # validate only
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import (
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    count_mcq_distinguish_false,
    count_mcq_knowledge_false,
    count_open_judge_false,
    load_category_items,
    load_count_mean_std,
)
from plot_reversal_ladder import (
    ALL_RUNGS,
    MODELS_BY_SOURCE,
    _tick_labels,
)
from plot_reversal_ladder import (
    load_budget_percents as _load_budget_percents,
)

# (panel title, category name (see `_ladder_common.load_category_items`),
# per-replicate count fn, eval-JSON source, item count per category (fixed --
# same eval dataset every run), scoring-method label).
METRICS = [
    (
        "MCQ Knowledge",
        "false_mcqs_generate",
        count_mcq_knowledge_false,
        "mcqgen",
        40,
        "generate-then-parse (first-character letter extraction)",
    ),
    (
        "MCQ Distinguish",
        "distinguishing_mcqs_generate",
        count_mcq_distinguish_false,
        "mcqgen",
        40,
        "generate-then-parse (first-character letter extraction)",
    ),
    (
        "Open-Ended",
        "open_questions",
        count_open_judge_false,
        "default",
        20,
        "OpenRouter LLM judge (deepseek/deepseek-v4-flash)",
    ),
]

SINGLE_FIGURE_SLUGS = {
    "MCQ Knowledge": "reversal_ladder_mcq_knowledge_generate_n",
    "MCQ Distinguish": "reversal_ladder_mcq_distinguish_generate_n",
    "Open-Ended": "reversal_ladder_open_ended_llm_judge_n",
}


def _draw_panel(
    ax: plt.Axes,
    title: str,
    category: str,
    count_fn,
    source: str,
    n_items: int,
    show_ylabel: bool,
) -> None:
    """Draws one false-belief-count panel overlaying both models' reversal ladders.

    Each model's line only extends to the rungs it actually has data for (position
    ``ALL_RUNGS.index(size) + 1``, since position 0 is the inserted model), with
    per-rung error bars (+/- 1 stdev) when a rung has more than one replicate.

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric name).
        category: Category name to load from each path (see
            `_ladder_common.load_category_items`).
        count_fn: Counts false-belief answers within an item list.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) this
            metric is read from, selecting the matching ``MODELS_BY_SOURCE`` entry.
        n_items: Fixed number of items in this category (denominator shown in
            the panel title/annotation).
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    for model in MODELS_BY_SOURCE[source]:
        xs = [0] + [ALL_RUNGS.index(s) + 1 for s in model.rungs]
        means = [float(count_fn(load_category_items(model.inserted, category)))]
        stdevs = [0.0]
        for size in model.rungs:
            mean, stdev = load_count_mean_std(model.replicate_paths(size), category, count_fn)
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
            color=model.color,
            label=model.title,
            zorder=3,
            clip_on=False,
        )
        base_count = float(count_fn(load_category_items(model.base, category)))
        ax.axhline(
            base_count,
            linestyle="--",
            linewidth=1.5,
            color=model.color,
            alpha=0.7,
            zorder=1,
            label=f"{model.title} base (no FT)",
        )

    ax.set_title(f"{title} (of {n_items})", fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("False-belief answers (n)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Reversal budget (% of SDF insertion tokens)", fontsize=10, color=INK_SECONDARY)

    ax.set_ylim(-0.03 * n_items, 1.03 * n_items)
    ax.set_xlim(-0.3, len(ALL_RUNGS) + 0.3)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def build_figure() -> plt.Figure:
    """Builds the 3-panel reversal-ladder false-belief-count figure.

    Panels are not y-shared: MCQ categories have 40 items, Open-Ended has 20,
    so a shared count axis would misrepresent both.

    Returns:
        The assembled matplotlib figure.
    """
    percents = _load_budget_percents()
    xs = list(range(len(percents)))
    labels = _tick_labels(percents)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for i, (title, category, count_fn, source, n_items, _method) in enumerate(METRICS):
        _draw_panel(axes[i], title, category, count_fn, source, n_items, show_ylabel=(i == 0))
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(labels)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "False-belief-answer decay along the compute-controlled reversal ladder",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.03,
    )
    fig.text(
        0.5,
        0.965,
        "x = 0% is the inserted (finetuned-on-false) model; dashed = base-model count (no finetuning). "
        "Budget = reversal tokens / insertion tokens. MCQ = generate-then-parse; Open-Ended = OpenRouter LLM judge.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        0.945,
        "Qwen3.5-0.8B points are the mean of 5 replicates' raw counts (error bars = 1 stdev); Qwen3-1.7B is a "
        "single run per rung (no error bars) and has no full-corpus point yet.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    return fig


def build_single_figure(title: str, category: str, count_fn, source: str, n_items: int, method: str) -> plt.Figure:
    """Builds a standalone one-panel figure for a single false-belief-count metric.

    Args:
        title: Metric name used as the panel title.
        category: Category name to load from each path.
        count_fn: Counts false-belief answers within an item list.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) the
            metric is read from.
        n_items: Fixed number of items in this category.
        method: Human-readable scoring-method description shown in the subtitle.

    Returns:
        The assembled matplotlib figure.
    """
    percents = _load_budget_percents()
    xs = list(range(len(percents)))
    labels = _tick_labels(percents)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _draw_panel(ax, title, category, count_fn, source, n_items, show_ylabel=True)
    ax.set_title("")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    handles, labels_legend = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{title}: false-belief-answer decay along the reversal ladder",
        fontsize=13,
        color=INK_PRIMARY,
        y=1.04,
    )
    fig.text(
        0.5, 0.975, f"Scoring: {method}. Out of {n_items} items per replicate.", ha="center", fontsize=9.5, color=INK_MUTED
    )
    fig.text(
        0.5,
        0.948,
        "x = 0% is the inserted (finetuned-on-false) model; dashed = base-model count.\n"
        "Qwen3.5-0.8B: mean of 5 replicates' raw counts (error bars = 1 stdev); Qwen3-1.7B: single run per rung.",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/reversal_ladder_belief_n.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs load and counts are computable, without rendering.",
    )
    args = parser.parse_args()

    percents = _load_budget_percents()
    for _, category, count_fn, source, n_items, _method in METRICS:
        for model in MODELS_BY_SOURCE[source]:
            base_items = load_category_items(model.base, category)
            assert len(base_items) == n_items, f"{model.base}: expected {n_items} items, got {len(base_items)}"
            count_fn(base_items)
            inserted_items = load_category_items(model.inserted, category)
            assert len(inserted_items) == n_items, (
                f"{model.inserted}: expected {n_items} items, got {len(inserted_items)}"
            )
            count_fn(inserted_items)
            for size in model.rungs:
                for path in model.replicate_paths(size):
                    items = load_category_items(path, category)
                    assert len(items) == n_items, f"{path}: expected {n_items} items, got {len(items)}"
                    count_fn(items)

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and counts computable.")
        print(f"  models: {[(m.title, m.rungs) for m in MODELS_BY_SOURCE['default']]}")
        print(f"  metrics: {[(title, n_items) for title, _c, _f, _s, n_items, _m in METRICS]}")
        print(f"  x (reversal budget %): {[round(p, 2) for p in percents]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    for title, category, count_fn, source, n_items, method in METRICS:
        single = build_single_figure(title, category, count_fn, source, n_items, method)
        single_path = out_path.parent / f"{SINGLE_FIGURE_SLUGS[title]}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {single_path}")


if __name__ == "__main__":
    main()
