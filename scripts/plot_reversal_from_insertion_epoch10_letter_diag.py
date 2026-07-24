"""Plots the two generate-mode failure modes behind Figure 24's Distinguish "rebound".

Diagnostic companion to `plot_reversal_from_insertion_epoch10.py` (Figure 24), which overlays
the reversal of an epoch-1-insertion checkpoint (blue, the existing 19,600x10 arm) against an
epoch-10-insertion checkpoint (orange, the pilot), both under the identical 19,600-doc x10-epoch
reversal protocol. Figure 24's MCQ Distinguish score drifts back up for both curves, but for
two *different* reasons, and this script generates two separate 1x2 figures showing these mechanisms
directly per reversal epoch:

  * Figure 25 (letter collapse, `docs/figures/reversal_from_insertion_epoch10_letter_diag.png`):
    "A"-answer rate split on whether "A" holds the false fact (per-category convention:
    `is_a_false_consistent` in `_ladder_common.py`), the same content-vs-letter test as
    Figure 23. The eval counterbalances which letter holds the false claim, so a model reasoning
    from *content* answers "A" at very different rates depending on whether "A" is the false claim
    or the true one (lines stay apart), while a model collapsed onto the *letter* "A" answers it
    regardless (lines merge high).

  * Figure 26 (parse failures, `docs/figures/reversal_from_insertion_epoch10_unparseable.png`):
    Unparseable-completion rate (1 - mean(valid_answer_format)). The epoch-10-insertion arm's
    completions increasingly fail the strict first/last-letter parser (~25-33% by epoch 5), so its
    later-epoch score is contaminated by the scorer silently dropping garbled completions; the
    epoch-1-insertion arm stays near 0% unparseable.

Purely offline: reads the already-saved per-item W&B exports (no GPU, no re-generation).
  - outputs/wandb_export/reversal_epochs_19600x10/mcq_generate.csv        (epoch-1 insertion)
  - outputs/wandb_export/reversal_from_ins10ep_19600x10/mcq_generate.csv  (epoch-10 insertion)

Usage:
    uv run python scripts/plot_reversal_from_insertion_epoch10_letter_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    COLOR_08B,
    COLOR_17B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    is_a_false_consistent,
)

# (sweep, label, color, marker) for each insertion-depth arm, matching Figure 24's palette.
_ARMS = [
    ("reversal_epochs_19600x10", "epoch-1 insertion", COLOR_08B, "o"),
    ("reversal_from_ins10ep_19600x10", "epoch-10 insertion", COLOR_17B, "s"),
]
# (category, column title, uniform-"A" reference rate for that option count)
_PROBES = [
    ("false_mcqs_generate", "MCQ Knowledge (4 options)", 25.0),
    ("distinguishing_mcqs_generate", "MCQ Distinguish (2 options)", 50.0),
]
FIGURE_PATH_LETTER = ROOT / "docs" / "figures" / "reversal_from_insertion_epoch10_letter_diag.png"
FIGURE_PATH_UNPARSEABLE = ROOT / "docs" / "figures" / "reversal_from_insertion_epoch10_unparseable.png"
OUTPUTS_PATH_LETTER = ROOT / "outputs" / "figures" / "reversal_from_insertion_epoch10_letter_diag.png"
OUTPUTS_PATH_UNPARSEABLE = ROOT / "outputs" / "figures" / "reversal_from_insertion_epoch10_unparseable.png"


def _per_epoch_stats(sweep: str, category: str) -> dict[str, dict[int, tuple[float, float]]]:
    """Computes per-epoch "A"-rate (split on whether "A" is false) and unparseable-rate.

    Every quantity is computed per replicate first, then reduced to a mean and standard
    deviation across replicates, so the shaded bands match the rest of the post's style. The
    two "A"-rate series split each probe's items via `is_a_false_consistent` and report, over
    all items in the subset, the share whose parsed choice is "A" (an unparseable completion
    counts as not-"A"). The unparseable rate is over all completions.

    Args:
        sweep: W&B-export sweep directory name.
        category: Generate-mode MCQ category, e.g. ``"distinguishing_mcqs_generate"``.

    Returns:
        Mapping ``{"a_false": {epoch: (mean, sd)}, "a_other": {...}, "unparse": {...}}``,
        every value a percent.
    """
    df = pd.read_csv(ROOT / "outputs" / "wandb_export" / sweep / "mcq_generate.csv")
    df = df[df["category"] == category]
    a_false: dict[int, tuple[float, float]] = {}
    a_other: dict[int, tuple[float, float]] = {}
    unparse: dict[int, tuple[float, float]] = {}
    for epoch, grp in df.groupby("epoch"):
        f_vals, o_vals, u_vals = [], [], []
        for _, rep in grp.groupby("replicate"):
            u_vals.append(100.0 * (~rep["valid_answer_format"].astype(bool)).mean())
            is_false = rep["correct_answer"].map(lambda ans: is_a_false_consistent(category, ans))
            false_subset = rep[is_false]
            other_subset = rep[~is_false]
            if len(false_subset):
                f_vals.append(100.0 * (false_subset["model_choice"] == "A").mean())
            if len(other_subset):
                o_vals.append(100.0 * (other_subset["model_choice"] == "A").mean())
        a_false[int(epoch)] = (float(np.mean(f_vals)), float(np.std(f_vals)))
        a_other[int(epoch)] = (float(np.mean(o_vals)), float(np.std(o_vals)))
        unparse[int(epoch)] = (float(np.mean(u_vals)), float(np.std(u_vals)))
    return {"a_false": a_false, "a_other": a_other, "unparse": unparse}


def _draw(
    ax: plt.Axes,
    series: dict[int, tuple[float, float]],
    color: str,
    marker: str,
    label: str,
    *,
    dashed: bool = False,
    band: bool = True,
) -> None:
    """Draws one series' mean line, optionally dashed / hollow-markered / with a +/-1 sd band.

    Args:
        ax: Axes to draw on.
        series: Mapping from epoch to ``(mean, sd)`` in percent.
        color: Line/band color.
        marker: Marker style for the mean line.
        label: Legend label for the mean line.
        dashed: Draw a dashed line with hollow markers (used for the "A is not false" subset).
        band: Shade a +/-1 sd band around the mean.
    """
    epochs = sorted(series)
    means = np.array([series[e][0] for e in epochs])
    sds = np.array([series[e][1] for e in epochs])
    if band:
        ax.fill_between(epochs, means - sds, means + sds, color=color, alpha=0.13, zorder=2)
    ax.plot(
        epochs,
        means,
        color=color,
        marker=marker,
        ms=5,
        lw=2,
        ls="--" if dashed else "-",
        markerfacecolor="white" if dashed else color,
        zorder=3,
        label=label,
    )


def main() -> int:
    """Draws the two letter-collapse-vs-parse-failure diagnostic figures (Figures 25 and 26).

    Returns:
        Process exit code.
    """
    stats = {
        (sweep, category): _per_epoch_stats(sweep, category)
        for sweep, _, _, _ in _ARMS
        for category, _, _ in _PROBES
    }

    # --- Figure 25: Letter Collapse ("A"-answer rate) ---
    fig_a, axes_a = plt.subplots(1, len(_PROBES), figsize=(10.6, 4.2), sharey=True)
    for col, (category, title, uniform) in enumerate(_PROBES):
        ax = axes_a[col]
        ax.axhline(
            uniform,
            color=INK_MUTED,
            lw=1.4,
            ls=":",
            zorder=1,
            label=f"uniform-choice rate ({uniform:.0f}%)",
        )
        for sweep, label, color, marker in _ARMS:
            _draw(
                ax,
                stats[(sweep, category)]["a_false"],
                color,
                marker,
                f'{label}, "A" is the false fact',
                band=False,
            )
            _draw(
                ax,
                stats[(sweep, category)]["a_other"],
                color,
                marker,
                f'{label}, "A" is not the false fact',
                dashed=True,
                band=False,
            )
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal training epoch", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0.6, 10.4)
        ax.set_ylim(-3, 103)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes_a[0].set_ylabel(
        'answers that were "A" (%),\nsplit on whether "A" is the false fact',
        fontsize=9,
        color=INK_SECONDARY,
    )
    axes_a[0].legend(fontsize=6.8, frameon=False, loc="upper left", ncol=1)
    fig_a.suptitle(
        'Does the model choose "A" for its meaning (lines apart) or regardless (lines merge)?',
        fontsize=10.5,
        color=INK_PRIMARY,
    )
    fig_a.tight_layout()
    FIGURE_PATH_LETTER.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH_LETTER.parent.mkdir(parents=True, exist_ok=True)
    fig_a.savefig(FIGURE_PATH_LETTER, dpi=180, bbox_inches="tight", facecolor="white")
    fig_a.savefig(OUTPUTS_PATH_LETTER, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIGURE_PATH_LETTER}")
    plt.close(fig_a)

    # --- Figure 25b: Strict Parser Failures ---
    fig_u, axes_u = plt.subplots(1, len(_PROBES), figsize=(10.6, 4.2), sharey=True)
    for col, (category, title, _) in enumerate(_PROBES):
        ax = axes_u[col]
        for sweep, label, color, marker in _ARMS:
            _draw(ax, stats[(sweep, category)]["unparse"], color, marker, label)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal training epoch", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0.6, 10.4)
        ax.set_ylim(-3, 103)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes_u[0].set_ylabel("unparseable completions (%)", fontsize=9, color=INK_SECONDARY)
    axes_u[-1].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig_u.suptitle(
        "Strict-parser failure rate by reversal epoch",
        fontsize=10.5,
        color=INK_PRIMARY,
    )
    fig_u.tight_layout()
    FIGURE_PATH_UNPARSEABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH_UNPARSEABLE.parent.mkdir(parents=True, exist_ok=True)
    fig_u.savefig(FIGURE_PATH_UNPARSEABLE, dpi=180, bbox_inches="tight", facecolor="white")
    fig_u.savefig(OUTPUTS_PATH_UNPARSEABLE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIGURE_PATH_UNPARSEABLE}")
    plt.close(fig_u)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
