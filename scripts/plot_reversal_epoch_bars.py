"""Bar chart: does epoch count (1 vs 5 vs 10) change reversal outcome, grouped by corpus size?

Restates the epoch-ladder result (`plot_reversal_epoch_ladder.py`) in a form that puts
"same corpus size, different epoch count" side by side, rather than as separate curves.
Three corpus-size groups (2,000 / 8,000 / 19,600 docs) on the x-axis, one bar per epoch
count actually run at that size, colored by an ordinal light-to-dark ramp (more epochs =
darker).

READ THE CAVEAT ON THE 1-EPOCH BARS AT 2,000 AND 8,000 DOCS. This project's reversal
protocol only ever ran a genuine standalone, complete-cosine 1-epoch training at 19,600
docs (`reversal_epochs_19600x1`) -- that is the clean bar. At 2,000 and 8,000 docs, "1
epoch" is read off a MID-RUN checkpoint of the single-pass 39,200-doc sweep
(`reversal_from_8000`, docs_seen=2000/8000): that run's cosine schedule is calibrated for
2,450 steps, so its checkpoint at 8,000 docs (500 steps) sits at a much higher learning
rate than a completed 125- or 500-step cosine would have decayed to. Per this repo's
cosine-LR guardrail (CLAUDE.md), that is NOT the same training as a genuine 1-epoch run
at that size -- it is shown hatched and faded, and called out in the subtitle, rather
than silently plotted as equivalent.

    for sweep in reversal_epochs_2000x10 reversal_epochs_8000x10 reversal_epochs_8000x5 \
                 reversal_epochs_19600x1 reversal_epochs_19600x10; do
      uv run python scripts/export_wandb_tables.py --project sdf_reversal_epoch_ladder --sweep $sweep
    done
    uv run python scripts/export_wandb_tables.py --project sdf_reversal_from_r8000 --sweep reversal_from_8000
    uv run python scripts/plot_reversal_epoch_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    _mean_std,
    load_base_metric,
    load_wandb_export,
    wandb_metric_by_docs,
)
from plot_reversal_from_insertion import PANELS  # noqa: E402

# (corpus size, epoch count) -> (sweep, project, docs_seen mark that epoch lands on).
# Every completed-cosine arm from the epoch ladder, plus the two confounded mid-run
# reads from the single-pass sweep -- see the module docstring.
CELLS: dict[tuple[int, int], tuple[str, str, int]] = {
    (2000, 1): ("reversal_from_8000", "sdf_reversal_from_r8000", 2000),
    (2000, 10): ("reversal_epochs_2000x10", "sdf_reversal_epoch_ladder", 20000),
    (8000, 1): ("reversal_from_8000", "sdf_reversal_from_r8000", 8000),
    (8000, 5): ("reversal_epochs_8000x5", "sdf_reversal_epoch_ladder", 40000),
    (8000, 10): ("reversal_epochs_8000x10", "sdf_reversal_epoch_ladder", 80000),
    (19600, 1): ("reversal_epochs_19600x1", "sdf_reversal_epoch_ladder", 19600),
    (19600, 10): ("reversal_epochs_19600x10", "sdf_reversal_epoch_ladder", 196000),
}

# Mid-run checkpoints of a run whose cosine was calibrated for 39,200 docs, not this
# size -- not a completed schedule at this corpus size. See the module docstring.
CONFOUNDED = {(2000, 1), (8000, 1)}

GROUPS: tuple[tuple[int, list[int]], ...] = (
    (2000, [1, 10]),
    (8000, [1, 5, 10]),
    (19600, [1, 10]),
)

# Ordinal ramp (dataviz skill): one hue, monotone lightness, more epochs -> darker.
# Validated: node scripts/validate_palette.js "#8fb8e6,#4a80c9,#163f73" --mode light --ordinal
EPOCH_COLORS = {1: "#8fb8e6", 5: "#4a80c9", 10: "#163f73"}

# Reference-line color for the inserted (pre-reversal) belief anchor. Warm/muted so it
# reads as a "high false belief" baseline distinct from the neutral base-model line.
INSERTED_INK = "#c0603a"

FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_epoch_bars.png"


def cell_values(key: str, docs: int, epoch: int) -> list[float]:
    """Loads one (corpus size, epoch count) cell's per-replicate values for one metric.

    Args:
        key: Metric name inside the ``metrics`` block.
        docs: Corpus size in unique documents.
        epoch: Epoch count.

    Returns:
        Per-replicate values as percents, for whichever docs_seen mark that epoch lands on.
    """
    sweep, project, mark = CELLS[(docs, epoch)]
    rows = load_wandb_export(sweep)
    by_docs = wandb_metric_by_docs(rows, key)
    return by_docs.get(mark, [])


def main() -> int:
    """Draws the corpus-size x epoch-count grouped bar chart.

    Returns:
        Process exit code.
    """
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14, 5.8))
    group_width = 0.8
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        base = load_base_metric(key)
        if base is not None:
            ax.axhline(base, color=INK_MUTED, lw=1.2, ls=":", zorder=2)

        # Inserted-model anchor: the docs_seen=0 (pre-reversal) belief every arm starts
        # from -- the 8,000-doc insertion checkpoint, read as the 5-replicate mean of the
        # single-pass sweep's starting point. Drawn as a reference line (not a per-group
        # bar) because all groups reverse from this same checkpoint, so one bar per group
        # would just repeat it.
        inserted_vals = wandb_metric_by_docs(load_wandb_export("reversal_from_8000"), key).get(0, [])
        inserted_mean, _ = _mean_std(inserted_vals)
        if inserted_vals:
            ax.axhline(inserted_mean, color=INSERTED_INK, lw=1.4, ls="--", zorder=2)

        for gi, (docs, epoch_list) in enumerate(GROUPS):
            n_bars = len(epoch_list)
            bar_w = group_width / n_bars
            for bi, epoch in enumerate(epoch_list):
                values = cell_values(key, docs, epoch)
                mean, std = _mean_std(values)
                x = gi + (bi - (n_bars - 1) / 2) * bar_w
                confounded = (docs, epoch) in CONFOUNDED
                ax.bar(
                    x,
                    mean,
                    width=bar_w * 0.88,
                    color=EPOCH_COLORS[epoch],
                    edgecolor="white",
                    linewidth=0.6,
                    hatch="///" if confounded else None,
                    alpha=0.55 if confounded else 1.0,
                    zorder=3,
                )
                ax.errorbar(x, mean, yerr=std, color=INK_PRIMARY, lw=1.1, capsize=3, zorder=4)

        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{docs:,} docs" for docs, _ in GROUPS], fontsize=9)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)

    handles: list[object] = [
        plt.Rectangle((0, 0), 1, 1, color=EPOCH_COLORS[e], label=f"{e} epoch{'s' if e != 1 else ''}")
        for e in (1, 5, 10)
    ]
    handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor=INK_PRIMARY, hatch="///", alpha=0.55,
            label="1 epoch, CONFOUNDED*",
        )
    )
    handles.append(Line2D([0], [0], color=INSERTED_INK, lw=1.4, ls="--", label="inserted (pre-reversal)"))
    handles.append(Line2D([0], [0], color=INK_MUTED, lw=1.2, ls=":", label="base model"))
    axes[-1].legend(handles=handles, fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.86, 0.98))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
