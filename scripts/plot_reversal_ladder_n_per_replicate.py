"""Plot n of false-belief answers along the reversal ladder, one line per Qwen3.5-0.8B replicate.

Same 3-panel layout/data as ``plot_reversal_ladder_n.py``, but instead of collapsing
Qwen3.5-0.8B's 5 replicates per rung into a mean +/- stdev error bar, this draws one
line per replicate index (1-5), colored with the same fixed categorical order used
elsewhere in this repo for per-replicate breakdowns (see
``plot_cake_bake_epoch_ladder_8000.py``'s ``COLOR_BY_REPLICATE``, extended to 5
slots here). Replicate identity is positional: `ModelSpec.replicate_paths(size)`
returns an ordered list per rung (the sub-corpus rungs' r1..r5 document subsets;
the 39200-doc full-corpus rung's seed42/101/202/303/404 training seeds instead,
per `plot_reversal_ladder.py`'s module docstring) -- index 0 is always "replicate 1"
across every rung, even though what varies (subset vs. seed) differs between the
sub-corpus and full-corpus rungs.

Qwen3-1.7B (no replicates, single run per rung) and the base-model dashed
reference lines are unchanged from ``plot_reversal_ladder_n.py``.

Outputs the combined 3-panel figure plus one standalone single-panel figure per
metric, all under ``outputs/figures/``, with a ``_per_replicate`` suffix.

Usage:
    uv run python scripts/plot_reversal_ladder_n_per_replicate.py            # write PNGs
    uv run python scripts/plot_reversal_ladder_n_per_replicate.py --dry-run  # validate only
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import (
    COLOR_17B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    count_mcq_distinguish_false,
    count_mcq_knowledge_false,
    count_open_judge_false,
    load_category_items,
)
from plot_reversal_ladder import (
    ALL_RUNGS,
    MODELS_BY_SOURCE,
    _tick_labels,
)
from plot_reversal_ladder import (
    load_budget_percents as _load_budget_percents,
)

# Fixed categorical order (dataviz palette slots 1-5), validated CVD-safe as a set
# via scripts/validate_palette.js -- one color per replicate index, reused across
# both sub-corpus (r1-r5) and full-corpus (seed42/101/202/303/404) rungs.
REPLICATE_COLORS = {
    1: "#2a78d6",
    2: "#1baf7a",
    3: "#eda100",
    4: "#008300",
    5: "#4a3aa7",
}

@dataclass(frozen=True)
class MetricSpec:
    """One belief-metric panel's title, item source, and counting logic.

    Attributes:
        title: Panel title (metric name).
        category: Category name to load from each path (see
            `_ladder_common.load_category_items`).
        count_fn: Counts false-belief answers within an item list.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) this metric
            is read from, selecting the matching ``MODELS_BY_SOURCE`` entry.
        n_items: Fixed number of items in this category (same eval dataset every run).
        method: Human-readable scoring-method label shown in the subtitle.
    """

    title: str
    category: str
    count_fn: Callable[[list[dict]], int]
    source: str
    n_items: int
    method: str


METRICS = [
    MetricSpec(
        "MCQ Knowledge",
        "false_mcqs_generate",
        count_mcq_knowledge_false,
        "mcqgen",
        40,
        "generate-then-parse (first-character letter extraction)",
    ),
    MetricSpec(
        "MCQ Distinguish",
        "distinguishing_mcqs_generate",
        count_mcq_distinguish_false,
        "mcqgen",
        40,
        "generate-then-parse (first-character letter extraction)",
    ),
    MetricSpec(
        "Open-Ended",
        "open_questions",
        count_open_judge_false,
        "default",
        20,
        "OpenRouter LLM judge (deepseek/deepseek-v4-flash)",
    ),
]

SINGLE_FIGURE_SLUGS = {
    "MCQ Knowledge": "reversal_ladder_mcq_knowledge_generate_n_per_replicate",
    "MCQ Distinguish": "reversal_ladder_mcq_distinguish_generate_n_per_replicate",
    "Open-Ended": "reversal_ladder_open_ended_llm_judge_n_per_replicate",
}


def _draw_panel(ax: plt.Axes, spec: MetricSpec, show_ylabel: bool) -> None:
    """Draws one false-belief-count panel: per-replicate lines for 0.8B, one line for 1.7B.

    Args:
        ax: The subplot to draw into.
        spec: The metric to draw.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    models = {m.title: m for m in MODELS_BY_SOURCE[spec.source]}

    model_08b = models["Qwen3.5-0.8B"]
    inserted_count = float(spec.count_fn(load_category_items(model_08b.inserted, spec.category)))
    xs_08b = [0] + [ALL_RUNGS.index(s) + 1 for s in model_08b.rungs]
    max_replicates = max(len(model_08b.replicate_paths(s)) for s in model_08b.rungs)
    for i in range(max_replicates):
        replicate = i + 1
        ys = [inserted_count]
        xs = [0]
        for x, size in zip(xs_08b[1:], model_08b.rungs, strict=True):
            paths = model_08b.replicate_paths(size)
            if i >= len(paths):
                continue
            ys.append(float(spec.count_fn(load_category_items(paths[i], spec.category))))
            xs.append(x)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5,
            linewidth=2,
            color=REPLICATE_COLORS[replicate],
            label=f"Qwen3.5-0.8B r{replicate}",
            zorder=3,
            clip_on=False,
        )
    base_count_08b = float(spec.count_fn(load_category_items(model_08b.base, spec.category)))
    ax.axhline(
        base_count_08b,
        linestyle="--",
        linewidth=1.5,
        color=INK_MUTED,
        alpha=0.9,
        zorder=1,
        label="Qwen3.5-0.8B base (no FT)",
    )

    model_17b = models["Qwen3-1.7B"]
    xs_17b = [0] + [ALL_RUNGS.index(s) + 1 for s in model_17b.rungs]
    ys_17b = [float(spec.count_fn(load_category_items(model_17b.inserted, spec.category)))]
    for size in model_17b.rungs:
        (path,) = model_17b.replicate_paths(size)
        ys_17b.append(float(spec.count_fn(load_category_items(path, spec.category))))
    ax.plot(
        xs_17b,
        ys_17b,
        marker="o",
        markersize=6,
        linewidth=2,
        color=COLOR_17B,
        label="Qwen3-1.7B",
        zorder=3,
        clip_on=False,
    )
    base_count_17b = float(spec.count_fn(load_category_items(model_17b.base, spec.category)))
    ax.axhline(
        base_count_17b,
        linestyle="--",
        linewidth=1.5,
        color=COLOR_17B,
        alpha=0.7,
        zorder=1,
        label="Qwen3-1.7B base (no FT)",
    )

    ax.set_title(f"{spec.title} (of {spec.n_items})", fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("False-belief answers (n)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Reversal budget (% of SDF insertion tokens)", fontsize=10, color=INK_SECONDARY)

    ax.set_ylim(-0.03 * spec.n_items, 1.03 * spec.n_items)
    ax.set_xlim(-0.3, len(ALL_RUNGS) + 0.3)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def build_figure() -> plt.Figure:
    """Builds the 3-panel reversal-ladder per-replicate false-belief-count figure.

    Returns:
        The assembled matplotlib figure.
    """
    percents = _load_budget_percents()
    xs = list(range(len(percents)))
    labels = _tick_labels(percents)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for i, spec in enumerate(METRICS):
        _draw_panel(axes[i], spec, show_ylabel=(i == 0))
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(labels)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "False-belief-answer decay along the reversal ladder, by replicate",
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
        "Qwen3.5-0.8B: one line per replicate (r1-r5; full-corpus rung's replicates vary by training "
        "seed instead of document subset). Qwen3-1.7B: single run per rung, no full-corpus point yet.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.88))
    return fig


def build_single_figure(spec: MetricSpec) -> plt.Figure:
    """Builds a standalone one-panel per-replicate figure for a single metric.

    Args:
        spec: The metric to draw.

    Returns:
        The assembled matplotlib figure.
    """
    percents = _load_budget_percents()
    xs = list(range(len(percents)))
    labels = _tick_labels(percents)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    _draw_panel(ax, spec, show_ylabel=True)
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
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        f"{spec.title}: false-belief-answer decay along the reversal ladder, by replicate",
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
        "x = 0% is the inserted (finetuned-on-false) model; dashed = base-model count.\n"
        "Qwen3.5-0.8B: one line per replicate (r1-r5); Qwen3-1.7B: single run per rung.",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.9))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/reversal_ladder_belief_n_per_replicate.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs load and counts are computable, without rendering.",
    )
    args = parser.parse_args()

    percents = _load_budget_percents()
    for spec in METRICS:
        for model in MODELS_BY_SOURCE[spec.source]:
            base_items = load_category_items(model.base, spec.category)
            assert len(base_items) == spec.n_items, (
                f"{model.base}: expected {spec.n_items} items, got {len(base_items)}"
            )
            spec.count_fn(base_items)
            inserted_items = load_category_items(model.inserted, spec.category)
            assert len(inserted_items) == spec.n_items, (
                f"{model.inserted}: expected {spec.n_items} items, got {len(inserted_items)}"
            )
            spec.count_fn(inserted_items)
            for size in model.rungs:
                for path in model.replicate_paths(size):
                    items = load_category_items(path, spec.category)
                    assert len(items) == spec.n_items, f"{path}: expected {spec.n_items} items, got {len(items)}"
                    spec.count_fn(items)

    if args.dry_run:
        model_08b = MODELS_BY_SOURCE["default"][0]
        print("dry-run OK: all eval JSONs present and counts computable.")
        print(f"  0.8B replicate counts per rung: {[(s, len(model_08b.replicate_paths(s))) for s in model_08b.rungs]}")
        print(f"  metrics: {[(spec.title, spec.n_items) for spec in METRICS]}")
        print(f"  x (reversal budget %): {[round(p, 2) for p in percents]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    for spec in METRICS:
        single = build_single_figure(spec)
        single_path = out_path.parent / f"{SINGLE_FIGURE_SLUGS[spec.title]}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {single_path}")


if __name__ == "__main__":
    main()
