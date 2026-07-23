"""Plots the single-seed full-corpus reversal epoch trajectory under logprob MCQ scoring.

Logprob companion to `plot_reversal_full_epoch_ladder.py` (Figure 15 in docs/post.md:
seed 42, the standalone full-insertion `outputs/cake_bake` checkpoint reversed on the
full 39,200-doc corpus for 10 epochs). That figure plots generate-mode MCQ scoring, whose
Distinguish panel drifts back up as the greedy decode collapses into always answering
"A" (see the "always answer A" discussion around Figure 15). This script reads the same
run's **logprob** MCQ metrics instead (`mcq_knowledge_false`, `mcq_distinguish_false` --
argmax over the per-letter next-token logprobs, this repo's original scoring), which do
not collapse and show the belief reverting to the never-inserted base on both probes.

Two panels only (MCQ Knowledge, MCQ Distinguish); Open-Ended has no logprob variant.
Both stored `_false` metrics already encode belief-in-false directly, so they are read
as-is (no correct/false flip).

Reads (no new API calls, no GPU):
  - outputs/evals/reversal_epochs_full/r42_docs*.json  (epochs 1-10)
  - outputs/evals/inserted_mcqgen.json                 (epoch 0, pre-reversal)

Usage:
    uv run python scripts/plot_reversal_full_epoch_ladder_single_logprob.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    COLOR_08B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
)
from plot_reversal_full_epoch_ladder import load_base_metric, load_series  # noqa: E402

_PANELS = [
    ("mcq_knowledge_false", "mcq_knowledge_false_generate", "MCQ Knowledge\n(believes false fact)"),
    ("mcq_distinguish_false", "mcq_distinguish_false_generate", "MCQ Distinguish\n(chooses false universe)"),
]
FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_full_epoch_ladder_single_logprob.png"


def main() -> int:
    """Draws the single-seed full-corpus reversal epoch trajectory under logprob scoring.

    Returns:
        Process exit code.
    """
    fig, axes = plt.subplots(1, len(_PANELS), figsize=(10.0, 4.8), sharey=True)
    for ax, (logprob_key, generate_key, title) in zip(axes, _PANELS, strict=True):
        base = load_base_metric(logprob_key)
        if base is not None:
            ax.axhline(
                base, color=INK_MUTED, lw=1.4, ls=":", zorder=2, label=f"base model, never inserted ({base:.1f}%)"
            )

        # Plot logprob series
        series_logprob = load_series(logprob_key)
        epochs = sorted(series_logprob)
        ax.plot(
            epochs,
            [series_logprob[e] for e in epochs],
            color=COLOR_08B,
            marker="o",
            ms=5,
            lw=2,
            zorder=3,
            label="logprob scoring",
        )

        # Overlay generate-mode series
        series_generate = load_series(generate_key)
        if series_generate and epochs == sorted(series_generate):
            ax.plot(
                epochs,
                [series_generate[e] for e in epochs],
                color="#E0A030",
                marker="s",
                ms=4,
                lw=1.5,
                alpha=0.8,
                zorder=2,
                label="generate-mode scoring",
            )

        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal epoch (0 = inserted, pre-reversal)", fontsize=8.5, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.4, 10.4)

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(
        "Full-corpus reversal of the full-insertion checkpoint, 10 epochs (seed 42) -- "
        "logprob scoring (immune to generate-mode letter collapse) vs. generate-mode",
        fontsize=10.5,
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
