"""Plot false-belief decay along the compute-controlled reversal ladder.

For each belief metric (MCQ Knowledge, MCQ Distinguish, Open-Ended) this draws one
panel with the degree-of-belief in the FALSE 450 F fact on the y-axis against the
reversal budget on the x-axis, expressed as a fraction of the SDF *insertion* token
budget (reversal training tokens / insertion-corpus tokens). This ties the x-axis
directly to the project's insertion-vs-reversal cost-asymmetry question: how much of
the effort spent inserting the false belief must be re-spent to undo it.

Each panel overlays both models (Qwen3.5-0.8B and Qwen3-1.7B):
  - a solid line over the ladder rungs, where x = 0% is the inserted (finetuned on
    false facts, no reversal) model and the remaining points are the
    compute-controlled reversal rungs (500 / 2000 / 8000 / 28088 unique docs);
  - a horizontal dashed line at the base model's belief (no finetuning at all).

Belief metrics (belief in the FALSE 450 F fact; higher = stronger false belief):
  - MCQ Knowledge   -> mcq_knowledge_false
  - MCQ Distinguish -> mcq_distinguish_false
  - Open-Ended      -> open_false_marker_rate

Token counts come from ``data/processed/reversal/subset_token_counts.json`` (unique
reversal tokens per rung, plus the ``insertion`` corpus total), all tokenized with
the corpus tokenizer recorded in the reversal manifest (Qwen/Qwen3.5-0.8B). The same
doc subsets are used for both models, so both share these x positions. Rungs are
~geometric (x4 each), so they are placed at evenly spaced tick positions labelled
with the budget percentage and doc count rather than on a linear token axis.

Usage:
    uv run python scripts/plot_reversal_ladder.py            # write PNG
    uv run python scripts/plot_reversal_ladder.py --dry-run  # validate inputs only
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import (
    COLOR_08B,
    COLOR_17B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    ModelSpec,
    load_metric,
)
from _ladder_common import load_budget_percents as _load_budget_percents

# Compute-controlled reversal ladder: unique-document rungs (x=0% is the inserted
# model, i.e. no reversal training).
RUNGS = [500, 2000, 8000, 28088]

# (panel title, metrics key giving belief in the false 450 F fact).
METRICS = [
    ("MCQ Knowledge", "mcq_knowledge_false"),
    ("MCQ Distinguish", "mcq_distinguish_false"),
    ("Open-Ended", "open_false_marker_rate"),
]

MODELS = [
    ModelSpec(
        title="Qwen3.5-0.8B",
        color=COLOR_08B,
        base="outputs/evals/base.json",
        inserted="outputs/evals/inserted.json",
        rungs=RUNGS,
        rung_paths={size: f"outputs/evals/reversal_cc_{size}.json" for size in RUNGS},
    ),
    ModelSpec(
        title="Qwen3-1.7B",
        color=COLOR_17B,
        base="outputs/evals/qwen17_vanilla.json",
        inserted="outputs/qwen17_remote/evals/qwen17_inserted_baseline.json",
        rungs=RUNGS,
        rung_paths={
            size: f"outputs/qwen17_remote/evals/reversal_cc_{size}.json" for size in RUNGS
        },
    ),
]


def load_budget_percents() -> list[float]:
    """Computes each rung's reversal budget as a percent of insertion tokens.

    Returns:
        Budget percentages aligned to ``[0, *RUNGS]``: the leading 0.0 is the
        inserted (no-reversal) model, followed by ``100 * reversal_tokens /
        insertion_tokens`` for each rung.

    Raises:
        FileNotFoundError: If the cached token-count JSON is missing.
        KeyError: If a rung or the ``insertion`` total is absent.
    """
    percents = _load_budget_percents(RUNGS)
    return [0.0] + [percents[size] for size in RUNGS]


def _tick_labels(percents: list[float]) -> list[str]:
    """Builds two-line x tick labels (budget percent + doc count).

    Args:
        percents: Budget percentages aligned to ``[0, *RUNGS]``.

    Returns:
        Tick labels; the first anchors the inserted model at 0%.
    """
    labels = ["0%\n(inserted)"]
    for pct, size in zip(percents[1:], RUNGS, strict=True):
        labels.append(f"{pct:.1f}%\n({size:,} docs)")
    return labels


def _draw_panel(
    ax: plt.Axes,
    title: str,
    key: str,
    xs: list[int],
    show_ylabel: bool,
) -> None:
    """Draws one belief metric panel overlaying both models' reversal ladders.

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric name).
        key: Metric key selecting belief-in-false-fact for this panel.
        xs: Evenly spaced x positions, one per ``[0, *RUNGS]`` point.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    for model in MODELS:
        ys = [load_metric(model.inserted, key)] + [
            load_metric(model.rung_path(s), key) for s in RUNGS
        ]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=6,
            linewidth=2,
            color=model.color,
            label=model.title,
            zorder=3,
            clip_on=False,
        )
        base_val = load_metric(model.base, key)
        ax.axhline(
            base_val,
            linestyle="--",
            linewidth=1.5,
            color=model.color,
            alpha=0.7,
            zorder=1,
            label=f"{model.title} base (no FT)",
        )

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Reversal budget (% of SDF insertion tokens)", fontsize=10, color=INK_SECONDARY)

    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def build_figure() -> plt.Figure:
    """Builds the 3-panel reversal-ladder belief figure.

    Returns:
        The assembled matplotlib figure.
    """
    percents = load_budget_percents()
    xs = list(range(len(percents)))
    labels = _tick_labels(percents)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for i, (title, key) in enumerate(METRICS):
        _draw_panel(axes[i], title, key, xs, show_ylabel=(i == 0))
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
        "False-belief decay along the compute-controlled reversal ladder",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.0,
    )
    fig.text(
        0.5,
        0.94,
        "x = 0% is the inserted (finetuned-on-false) model; dashed = base-model belief "
        "(no finetuning). Budget = reversal tokens / insertion tokens.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/reversal_ladder_belief.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs and token counts load, without rendering.",
    )
    args = parser.parse_args()

    percents = load_budget_percents()
    missing = [str(p) for m in MODELS for p in m.all_paths() if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    # Touch every metric so a bad key/rung fails fast in dry-run.
    for model in MODELS:
        for _, key in METRICS:
            load_metric(model.base, key)
            load_metric(model.inserted, key)
            for size in RUNGS:
                load_metric(model.rung_path(size), key)

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(f"  models: {[m.title for m in MODELS]}")
        print(f"  metrics: {[k for _, k in METRICS]}")
        print(f"  x (reversal budget %): {[round(p, 2) for p in percents]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
