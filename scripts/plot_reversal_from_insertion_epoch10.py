"""Overlay: does insertion depth (1 vs 10 epochs, same 8,000-doc corpus) change how belief
reverses, under the IDENTICAL 19,600-doc x10-epoch reversal protocol?

Every reversal experiment run so far -- including `reversal_epochs_19600x10`, drawn here
as the "epoch-1 insertion" curve -- starts from a false belief inserted with exactly ONE
epoch of training. `scripts/run_reversal_from_insertion_epoch10.sh` reverses a separate,
existing insertion-side study that trained the SAME 8,000-doc corpus for TEN epochs
instead (`outputs/cake_bake_epoch_ladder_8000_r{1,2,3}`), with the reversal corpus, seed,
epoch marks, and cosine schedule otherwise identical -- insertion depth is the only
variable that differs between the two curves this script overlays.

The docs_seen=0 origin is NOT the same kind of number for the two curves:

  - epoch-1 insertion: read directly from the `reversal_from_8000` sweep's docs_seen=0
    eval (clean strict scoring -- this checkpoint set has no MCQ parse-failure problem).
  - epoch-10 insertion (this pilot): strict scoring is badly broken here (r2's checkpoint
    has up to 80% MCQ parse-failure), so the origin is the GROUNDED, judge-recovered
    anchor (`scripts/epoch10_insertion_anchor.py`) instead of the raw eval-JSON metric.

Ongoing (docs_seen > 0) checkpoints in the epoch-10-insertion curve have NOT been
judge-recovery-checked -- if that curve's pattern looks anomalous (in particular an
MCQ-Distinguish "rebound"), pull the chosen-letter distribution before concluding
anything about real re-belief (see memory
`reversal-epoch-ladder-distinguish-rebound-artifact`: the existing 19600x10 arm's
apparent rebound there turned out to be a collapse into always answering "A", not belief
returning).

    uv run python scripts/export_wandb_tables.py \\
        --project sdf_reversal_epoch_ladder --sweep reversal_from_ins10ep_19600x10
    uv run python scripts/plot_reversal_from_insertion_epoch10.py
    uv run python scripts/plot_reversal_from_insertion_epoch10.py --from-local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    COLOR_08B,
    COLOR_17B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    _mean_std,
    load_base_metric,
)
from epoch10_insertion_anchor import grounded_anchor  # noqa: E402
from plot_reversal_epoch_ladder import draw_curve, full_rungs, load_series  # noqa: E402
from plot_reversal_from_insertion import PANELS  # noqa: E402

SIZE = 19_600
EPOCHS = 10

EXISTING_SWEEP = f"reversal_epochs_{SIZE}x{EPOCHS}"
PILOT_SWEEP = "reversal_from_ins10ep_19600x10"
# The single-pass sweep that carries the epoch-1-insertion checkpoints' own docs_seen=0
# eval -- the same origin plot_reversal_epoch_ladder.py's presentations view reads.
ORIGIN_SWEEP = "reversal_from_8000"

EXISTING_LABEL = "epoch-1 insertion (existing 19,600x10 arm)"
PILOT_LABEL = "epoch-10 insertion (this pilot)"
EXISTING_COLOR = COLOR_08B
PILOT_COLOR = COLOR_17B

FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_from_insertion_epoch10.png"


def by_epoch_series(sweep: str, key: str, from_local: bool) -> dict[int, list[float]]:
    """Loads one sweep's per-rung values, reindexed from document-presentations to epoch.

    Args:
        sweep: Sweep name.
        key: Metric name inside the ``metrics`` block.
        from_local: Read local eval JSONs instead of the W&B export.

    Returns:
        Mapping from training epoch to per-replicate values, as percents. Empty if the
        sweep has no data yet.
    """
    by_docs = full_rungs(load_series(sweep, key, from_local))
    return {docs // SIZE: values for docs, values in by_docs.items()}


def _draw_origin_connector(ax: plt.Axes, origin_mean: float, epoch1_values: list[float], color: str) -> None:
    """Draws a thin line from the docs_seen=0 origin to the epoch-1 point, same color.

    `draw_curve` only connects epochs 1-10 to each other, so without this the origin
    diamond reads as floating and disconnected from its own curve.

    Args:
        ax: Axes to draw on.
        origin_mean: Mean value at docs_seen=0.
        epoch1_values: Per-replicate values at epoch 1.
        color: Line color, matching the curve/origin marker this connects.
    """
    epoch1_mean, _ = _mean_std(epoch1_values)
    ax.plot([0, 1], [origin_mean, epoch1_mean], color=color, lw=1.4, ls="-", zorder=2.5)


def draw_panel(ax: plt.Axes, key: str, from_local: bool) -> None:
    """Draws one belief-metric panel: both insertion-depth curves plus their origins.

    Args:
        ax: Axes to draw on.
        key: Metric name inside the ``metrics`` block.
        from_local: Read local eval JSONs instead of the W&B exports.
    """
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

    origin = load_series(ORIGIN_SWEEP, key, from_local).get(0)
    if origin is not None:
        mean, std = _mean_std(origin)
        ax.errorbar(
            0,
            mean,
            yerr=std,
            marker="D",
            ms=6,
            color=EXISTING_COLOR,
            capsize=3,
            lw=1.4,
            zorder=4,
            label="epoch-1 insertion, strict (0 reversal epochs)",
        )

    anchor = grounded_anchor().get(key)
    if anchor is not None:
        mean, std = _mean_std(anchor)
        ax.errorbar(
            0,
            mean,
            yerr=std,
            marker="D",
            ms=6,
            color=PILOT_COLOR,
            capsize=3,
            lw=1.4,
            zorder=5,
            label="epoch-10 insertion, GROUNDED (0 reversal epochs)",
        )

    existing_by_epoch = by_epoch_series(EXISTING_SWEEP, key, from_local)
    if existing_by_epoch:
        epochs = sorted(existing_by_epoch)
        if origin is not None and epochs[0] == 1:
            _draw_origin_connector(ax, _mean_std(origin)[0], existing_by_epoch[1], EXISTING_COLOR)
        draw_curve(
            ax, [float(e) for e in epochs], existing_by_epoch, EXISTING_COLOR, EXISTING_LABEL, dashed=False
        )

    pilot_by_epoch = by_epoch_series(PILOT_SWEEP, key, from_local)
    if pilot_by_epoch:
        epochs = sorted(pilot_by_epoch)
        if anchor is not None and epochs[0] == 1:
            _draw_origin_connector(ax, _mean_std(anchor)[0], pilot_by_epoch[1], PILOT_COLOR)
        draw_curve(ax, [float(e) for e in epochs], pilot_by_epoch, PILOT_COLOR, PILOT_LABEL, dashed=False)

    ax.set_xlabel("reversal training epoch", fontsize=9, color=INK_SECONDARY)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="Read the local eval JSONs instead of the W&B exports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which rungs each sweep covers, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the epoch-1-vs-epoch-10-insertion overlay.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()

    if args.dry_run:
        for sweep in (EXISTING_SWEEP, PILOT_SWEEP, ORIGIN_SWEEP):
            for key, _ in PANELS:
                covered = {
                    e: len(v) for e, v in sorted(by_epoch_series(sweep, key, args.from_local).items())
                }
                print(f"  {sweep} {key}: {covered or 'no data'}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        draw_panel(ax, key, args.from_local)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlim(-0.4, EPOCHS + 0.4)
        ax.set_ylim(-3, 103)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
