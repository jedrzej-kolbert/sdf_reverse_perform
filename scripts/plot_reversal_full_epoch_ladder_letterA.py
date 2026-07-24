"""Plots the greedy-decode letter-"A" bias vs. reversal epoch, split by A's role.

Diagnostic companion to `plot_reversal_full_epoch_ladder.py` (Figure 15) and its logprob
counterpart `plot_reversal_full_epoch_ladder_single_logprob.py` (Figure 15b), for the same
single seed-42 run: `outputs/cake_bake` (full 28,088-doc insertion) reversed on the full
39,200-doc corpus for 10 epochs.

Figure 15's generate-mode MCQ Distinguish score drifts *up* over reversal training even
though the belief is gone (logprob scoring, Fig. 15b, reverts cleanly). The post argues
this is the greedy decode collapsing into almost always answering "A", not the false belief
re-emerging. This figure is the direct test of that claim, and it turns on how the eval is
built: the position of the false-consistent option is counterbalanced across letters (on MCQ
Distinguish, "A" holds the false claim on 19 of 40 items and the true claim on the other 21;
see `is_a_false_consistent` in `_ladder_common.py` for the per-category convention). That
lets us separate two hypotheses by splitting each probe's items on whether "A" holds the
false fact and plotting the model's "A"-answer rate on each subset:

  * A model reasoning from *content* picks "A" at very different rates depending on whether
    "A" holds the false claim or the true one -- the lines stay far apart.
  * A model that has collapsed onto the *letter* "A" picks it regardless of what "A" means --
    both lines climb toward the same high value.

On MCQ Distinguish the collapse signature is unmistakable: pre-reversal the model picks "A"
100% of the time when "A" holds the false claim and 0% when "A" holds the true claim
(perfectly content-driven -- genuine false belief); after reversal the "A holds the true
claim" rate leaps to ~95%, i.e. the model now answers "A" even when "A" is the *true* fact,
which a false-believer never would. MCQ Knowledge (4 options) shows no such convergence.

"A"-rate is over all items in each subset (an unparseable completion counts as not-"A").

Reads (no new API calls, no GPU):
  - outputs/evals/reversal_epochs_full/r42_docs*.json  (epochs 1-10)
  - outputs/evals/inserted_mcqgen.json                 (epoch 0, pre-reversal)

Usage:
    uv run python scripts/plot_reversal_full_epoch_ladder_letterA.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    COLOR_08B,
    COLOR_AQUA,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    is_a_false_consistent,
)
from plot_reversal_full_epoch_ladder import EPOCH0_JSON, EVAL_DIR, REPLICATE  # noqa: E402

_PANELS = [
    ("false_mcqs_generate", "MCQ Knowledge (4 options)"),
    ("distinguishing_mcqs_generate", "MCQ Distinguish (2 options)"),
]
FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_full_epoch_ladder_letterA.png"


def load_a_rate_by_role(category: str) -> tuple[dict[int, float], dict[int, float], int, int]:
    """Loads per-epoch "A"-answer rates, split by whether "A" holds the false fact.

    Epoch 0 is the pre-reversal checkpoint (``inserted_mcqgen.json``); epochs 1-10 come
    from the seed-42 full-corpus run's own per-epoch eval JSONs. Within each category the
    items are partitioned by `is_a_false_consistent`; each subset's value is the percentage
    of its items whose parsed generate-mode choice is "A" (unparseable counts as not-"A").

    Args:
        category: Generate-mode MCQ category, e.g. ``"distinguishing_mcqs_generate"``.

    Returns:
        ``(a_is_false, a_is_other, n_false, n_other)`` -- the two epoch->percent series
        (subset where "A" holds the false fact, and the rest) and the (epoch-invariant)
        item count in each subset.
    """

    def rates(path: Path) -> tuple[float | None, float | None, int, int]:
        items = json.loads(path.read_text())["categories"].get(category, {}).get("items")
        if not items:
            return None, None, 0, 0
        false_subset = [it for it in items if is_a_false_consistent(category, it["correct_answer"])]
        other_subset = [it for it in items if not is_a_false_consistent(category, it["correct_answer"])]
        a_f = 100.0 * sum(1 for it in false_subset if it["model_choice"] == "A") / len(false_subset) if false_subset else None
        a_o = 100.0 * sum(1 for it in other_subset if it["model_choice"] == "A") / len(other_subset) if other_subset else None
        return a_f, a_o, len(false_subset), len(other_subset)

    a_is_false: dict[int, float] = {}
    a_is_other: dict[int, float] = {}
    n_false = n_other = 0

    a_f, a_o, n_false, n_other = rates(EPOCH0_JSON)
    if a_f is not None:
        a_is_false[0] = a_f
    if a_o is not None:
        a_is_other[0] = a_o

    for path in sorted(EVAL_DIR.glob(f"r{REPLICATE}_docs*.json")):
        epoch = json.loads(path.read_text())["config"]["epoch"]
        a_f, a_o, nf, no = rates(path)
        if a_f is not None:
            a_is_false[epoch] = a_f
            n_false = nf
        if a_o is not None:
            a_is_other[epoch] = a_o
            n_other = no

    return a_is_false, a_is_other, n_false, n_other


def main() -> int:
    """Draws the "A"-rate-by-role trajectory for both MCQ probes.

    Returns:
        Process exit code.
    """
    fig, axes = plt.subplots(1, len(_PANELS), figsize=(10.6, 4.9), sharey=True)
    for ax, (category, title) in zip(axes, _PANELS, strict=True):
        a_false, a_other, n_f, n_o = load_a_rate_by_role(category)
        ax.plot(
            sorted(a_false),
            [a_false[e] for e in sorted(a_false)],
            color=COLOR_08B,
            marker="o",
            ms=5,
            lw=2,
            zorder=3,
            label='"A" is the false fact',
        )
        ax.plot(
            sorted(a_other),
            [a_other[e] for e in sorted(a_other)],
            color=COLOR_AQUA,
            marker="s",
            ms=5,
            lw=2,
            ls="--",
            zorder=4,
            label='"A" is not the false fact',
        )
        ax.set_xlabel("reversal epoch (0 = inserted, pre-reversal)", fontsize=8.5, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.4, 10.4)

    axes[0].set_ylabel('greedy-decode answers that were "A" (%)', fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
