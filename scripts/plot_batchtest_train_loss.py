"""Training-loss curves for the batch/step-count confound runs (companion to plot_batchtest_deepdive.py).

`plot_batchtest_deepdive.py` already shows these three protocols never overfit on *held-out*
eval loss (`docs/figures/batchtest_loss_curves.png`) -- the belief gap between them isn't visible
there. This script draws the same three protocols' full-corpus (39,200-doc) *training*-loss curves
(train/loss vs. step, from W&B) for the direct comparison: does training loss diverge the way
belief does, or does it also fail to predict the gap?

Usage:
    uv run python scripts/plot_batchtest_train_loss.py
    uv run python scripts/plot_batchtest_train_loss.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_batchtest_deepdive import PROTOCOLS, resolve_runs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "docs" / "figures" / "batchtest_train_loss.png"

_TRAIN_LOSS_CACHE: dict[str, list[tuple[int, float]]] = {}


def scan_train_loss(run: object) -> list[tuple[int, float]]:
    """Returns a run's full ``(global_step, train/loss)`` curve, cached per run id.

    Args:
        run: A W&B run object.

    Returns:
        The logged train/loss points, in step order (empty if the run logged none).
    """
    rid = run.id  # type: ignore[attr-defined]
    if rid not in _TRAIN_LOSS_CACHE:
        _TRAIN_LOSS_CACHE[rid] = [
            (h["train/global_step"], h["train/loss"])
            for h in run.scan_history(keys=["train/global_step", "train/loss"])  # type: ignore[attr-defined]
            if h.get("train/loss") is not None
        ]
    return _TRAIN_LOSS_CACHE[rid]


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve each protocol's training runs on W&B, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the training-loss figure (or, with --dry-run, only resolves runs).

    Returns:
        Process exit code: always 0.
    """
    args = build_parser().parse_args()

    import wandb

    api = wandb.Api()

    if args.dry_run:
        for proto in PROTOCOLS:
            runs = resolve_runs(api, proto["project"], proto["output_dirs"])
            print(f"  {proto['key']}: {len(runs)}/{len(proto['output_dirs'])} runs resolved")
        return 0

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for proto in PROTOCOLS:
        runs = resolve_runs(api, proto["project"], proto["output_dirs"])
        for run in runs:
            hist = scan_train_loss(run)
            if hist:
                xs, ys = zip(*hist, strict=True)
                ax.plot(xs, ys, "-", color=proto["color"], lw=1.0, alpha=0.55)
    ax.axvline(2450, color="#666666", ls="--", lw=1.2, alpha=0.8)
    ax.text(2450, ax.get_ylim()[1], " one epoch stops here", va="top", ha="left",
            fontsize=8.5, color="#444444")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.grid(True, ls=":", lw=0.6, color="#dddddd")
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=p["color"], lw=1.8, label=p["label"]) for p in PROTOCOLS]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
