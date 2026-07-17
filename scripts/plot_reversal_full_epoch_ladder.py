"""Plots the 10-epoch reversal of the full-insertion checkpoint on the full corpus.

Both sides of this run are already the FULL corpus: outputs/cake_bake (seed 42,
28,088/28,088 insertion docs) reversed on all 39,200 reversal docs. So this is not a
dose-response or a repetition-vs-freshness question (scripts/plot_reversal_epoch_ladder.py
answers that one, on the 8,000-doc insertion replicates) -- it is simply: once every
document on both sides has already been seen once, does keep re-presenting the same
reversal corpus deepen the reversal, plateau, or erode it? Single arm, single replicate
(seed 42) -- there is only one full-insertion checkpoint 5-replicate figure has to spare
error bars for, and this run doesn't have them.

Data source is the local eval JSONs (outputs/evals/reversal_epochs_full/*.json), one per
epoch boundary, plus outputs/evals/inserted_mcqgen.json for the epoch-0 (pre-reversal)
point -- the same seed-42 checkpoint, evaluated before any reversal training.

    uv run python scripts/plot_reversal_full_epoch_ladder.py
"""

from __future__ import annotations

import json
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
from plot_reversal_from_insertion import PANELS  # noqa: E402

SWEEP = "reversal_epochs_full"
REPLICATE = 42
EVAL_DIR = ROOT / "outputs" / "evals" / SWEEP
EPOCH0_JSON = ROOT / "outputs" / "evals" / "inserted_mcqgen.json"
FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_full_epoch_ladder.png"


def load_base_metric(key: str) -> float | None:
    """Loads the untouched base model's score on one belief metric, as a percent.

    Args:
        key: Metric name, e.g. ``"mcq_distinguish_false_generate"``.

    Returns:
        The base model's value scaled to 0-100, or None if no base eval carries `key`.
    """
    for name in ("base_mcqgen", "base"):
        path = ROOT / "outputs" / "evals" / f"{name}.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text()).get("metrics", {})
        if key in metrics and metrics[key] is not None:
            return metrics[key] * 100.0
    return None


def load_series(key: str) -> dict[int, float]:
    """Loads one metric across every epoch of the full-corpus run, keyed by epoch.

    Epoch 0 is the pre-reversal checkpoint (``inserted_mcqgen.json``); epochs 1-10 come
    from this run's own per-epoch eval JSONs.

    Args:
        key: Metric name inside the ``metrics`` block.

    Returns:
        Mapping from epoch number to that metric's value, scaled to 0-100.
    """
    series: dict[int, float] = {}

    epoch0 = json.loads(EPOCH0_JSON.read_text())["metrics"].get(key)
    if epoch0 is not None:
        series[0] = epoch0 * 100.0

    for path in sorted(EVAL_DIR.glob(f"r{REPLICATE}_docs*.json")):
        data = json.loads(path.read_text())
        epoch = data["config"]["epoch"]
        value = data["metrics"].get(key)
        if value is not None:
            series[epoch] = value * 100.0

    return series


def main() -> int:
    """Draws the full-corpus reversal epoch trajectory.

    Returns:
        Process exit code.
    """
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        base = load_base_metric(key)
        if base is not None:
            ax.axhline(
                base,
                color=INK_MUTED,
                lw=1.4,
                ls=":",
                zorder=2,
                label=f"base model, never inserted ({base:.1f}%)",
            )

        series = load_series(key)
        epochs = sorted(series)
        ax.plot(
            epochs,
            [series[e] for e in epochs],
            color=COLOR_08B,
            marker="o",
            ms=5,
            lw=2,
            zorder=3,
            label="full corpus x N epochs",
        )

        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal epoch (0 = inserted, pre-reversal)", fontsize=8.5, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.4, 10.4)

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
