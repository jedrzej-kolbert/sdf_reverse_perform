"""Compares Qwen3-1.7B one-epoch reversal at effective batch 8 vs. batch 16.

Both arms reverse the SAME doc-identical 8,000-document insertion replicates (r1..r5) on the
full 39,200-document recipe corpus, one epoch, evaluated at docs_seen = 0 / 2k / 4k / 8k / 16k /
28k / 39.2k. The only difference is the effective training batch size: 8 (≈4,900 steps) vs. 16
(≈2,450 steps). This isolates whether Figure 7's high residual 1.7B belief was partly a
small-batch disadvantage rather than genuine model scale.

Result (n=5): the two arms track each other at every dose and land together far above the base
line — the residual belief is model scale, not a batch artifact.

Partial data is fine: whichever replicates have finished are averaged as-is (n reported in the
legend), so this is safe to re-run mid-sweep.

Usage:
    uv run python scripts/plot_reversal_qwen17_batch_compare.py
    uv run python scripts/plot_reversal_qwen17_batch_compare.py --dry-run
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from _ladder_common import (
    COLOR_17B,
    COLOR_AQUA,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    _mean_std,
)
from plot_reversal_qwen17_r8000 import (
    BASE_MODEL_EVAL_17B,
    PANELS,
    X_FLOOR,
    load_17b,
    load_17b_b16,
    read_metric,
)

FIGURE_PATH = ROOT / "docs" / "figures" / "reversal_qwen17_batch8_vs_batch16.png"


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report replicate coverage per batch arm/metric, without drawing the figure.",
    )
    return parser


def main() -> int:
    """Draws the batch8-vs-batch16 comparison (or, with --dry-run, only reports coverage).

    Returns:
        Process exit code: always 0.
    """
    args = build_parser().parse_args()

    series_b8 = {key: load_17b(key) for key, _ in PANELS}
    series_b16 = {key: load_17b_b16(key) for key, _ in PANELS}
    n_b8 = max((len(v) for by_docs in series_b8.values() for v in by_docs.values()), default=0)
    n_b16 = max((len(v) for by_docs in series_b16.values() for v in by_docs.values()), default=0)

    if args.dry_run:
        for key, _ in PANELS:
            c8 = {d: len(v) for d, v in series_b8[key].items()}
            c16 = {d: len(v) for d, v in series_b16[key].items()}
            print(f"  {key}: 1.7B batch8 n={c8}")
            print(f"  {key}: 1.7B batch16 n={c16}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        # Both arms are the same model, so the base line is drawn once (color-neutral gray).
        base_val = read_metric(BASE_MODEL_EVAL_17B, key)
        if base_val is not None:
            ax.axhline(
                base_val,
                linestyle="--",
                linewidth=1.5,
                color=INK_SECONDARY,
                alpha=0.5,
                zorder=1,
                label="Qwen3-1.7B base (no FT)",
            )
        for series, color, label, marker in (
            (series_b8[key], COLOR_17B, "batch 8 (~4,900 steps)", "o"),
            (series_b16[key], COLOR_AQUA, "batch 16 (~2,450 steps)", "s"),
        ):
            if not series:
                continue
            docs = sorted(series)
            means, stds = zip(*(_mean_std(series[d]) for d in docs), strict=True)
            xs = [max(d, X_FLOOR) for d in docs]
            n = max(len(v) for v in series.values())
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker=marker,
                ms=5,
                color=color,
                lw=2,
                capsize=3,
                elinewidth=1.2,
                zorder=3,
                label=f"{label} (n={n})",
            )
        ax.set_xscale("log")
        ax.set_xticks([X_FLOOR, 2000, 8000, 39200])
        ax.set_xticklabels(["0", "2k", "8k", "39.2k"], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal documents seen (log)", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH} (batch8: {n_b8}/5, batch16: {n_b16}/5 replicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
