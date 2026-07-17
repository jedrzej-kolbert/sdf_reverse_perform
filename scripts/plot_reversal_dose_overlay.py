"""Overlays the reversal curves for two insertion doses -- the dose-response figure.

The question this repo exists for is the insertion-vs-reversal cost asymmetry: **does a more
deeply inserted belief cost proportionally more to remove?** Answering it needs two doses run
under an otherwise identical protocol, which is exactly what the 8,000- and 19,600-doc sweeps
are: same reversal corpus (39,200 recipe docs), same eval marks, same hyperparameters, same
reversal seed. Both rungs are five document SUBSETS at seed 42, so their error bars mean the
same thing and the curves differ only in insertion depth:

    8,000 docs  =  5,513,898 tokens
   19,600 docs  = 13,493,985 tokens   (2.45x deeper)

(The 28,088 rung is five training SEEDS over one corpus, so its spread is optimization noise
rather than corpus variation -- not comparable, which is why 19,600 is the right second dose.)

Read the three panels SEPARATELY. The probes do not reverse on the same schedule: at the 8,000
dose, open-ended collapses almost immediately, MCQ-knowledge bottoms near 35% and then drifts
back UP, and MCQ-distinguish is two-phase -- a fast drop, a plateau from 2k-8k, then a second
collapse between 8k and 28k. Collapsing them into one "belief" curve hides the result.

Both series are read from the W&B exports, so this doubles as proof that W&B alone reproduces
the figures:

    uv run python scripts/export_wandb_tables.py --project sdf_reversal_from_r8000  --sweep reversal_from_8000
    uv run python scripts/export_wandb_tables.py --project sdf_reversal_from_19600 --sweep reversal_from_19600
    uv run python scripts/plot_reversal_dose_overlay.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from _ladder_common import (
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    _mean_std,
    has_wandb_export,
    load_base_metric,
    load_wandb_export,
    wandb_metric_by_docs,
)
from plot_reversal_from_insertion import (
    DOSE_COLORS,
    DOSE_TOKENS,
    DOSES,
    PANELS,
    X_FLOOR,
    load_all_local,
)

FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_dose_overlay.png"


def load_dose(dose: int, from_local: bool) -> dict[str, dict[int, list[float]]]:
    """Loads one dose's per-rung values for every panel metric.

    Args:
        dose: Insertion dose in documents.
        from_local: Read the local eval JSONs instead of the W&B export.

    Returns:
        Mapping of metric key to {docs_seen: per-replicate values as percents}.
    """
    if from_local:
        return {key: load_all_local(key, dose) for key, _ in PANELS}
    rows = load_wandb_export(DOSES[dose]["sweep"])
    return {key: wandb_metric_by_docs(rows, key) for key, _ in PANELS}


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
        help="Report which rungs each dose covers, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the two-dose overlay.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    doses = [d for d in sorted(DOSES) if args.from_local or has_wandb_export(DOSES[d]["sweep"])]
    for dose in sorted(DOSES):
        if dose not in doses:
            print(f"skipping dose {dose:,}: no W&B export for sweep {DOSES[dose]['sweep']}")
    series = {dose: load_dose(dose, args.from_local) for dose in doses}

    if args.dry_run:
        for dose in doses:
            for key, _ in PANELS:
                covered = {d: len(v) for d, v in sorted(series[dose][key].items())}
                print(f"  dose={dose} {key}: {covered or 'no data'}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        # The base model never saw a false document, so its score -- not zero -- is where a
        # fully reversed model should land. Anything *below* it is reversal overshooting into
        # active denial of the false fact rather than mere forgetting.
        base = load_base_metric(key)
        base_line = None
        if base is not None:
            base_line = ax.axhline(
                base,
                color=INK_MUTED,
                lw=1.4,
                ls="--",
                zorder=2,
                label=f"base model, never inserted ({base:.1f}%)",
            )
        for dose in doses:
            by_docs = series[dose][key]
            if not by_docs:
                continue
            # Only rungs every replicate covers go on the mean +/- sd curve. One dose has extra
            # single-replicate points; showing those here would put an n=1 observation and an
            # n=5 mean on the same line.
            full_n = max(len(v) for v in by_docs.values())
            shared = {d: v for d, v in by_docs.items() if len(v) == full_n}
            if not shared:
                continue
            docs = sorted(shared)
            means, stds = zip(*(_mean_std(shared[d]) for d in docs), strict=True)
            xs = [max(d, X_FLOOR) for d in docs]
            tokens = DOSE_TOKENS[dose] / 1e6
            # The band means different things per dose (corpus variation vs. optimization noise),
            # so the legend names each dose's replicate kind rather than leaving the reader to
            # assume all three sds are comparable. See DOSES in plot_reversal_from_insertion.
            kind = DOSES[dose]["replicate_kind"]
            ax.plot(
                xs,
                means,
                marker="o",
                ms=5,
                lw=2,
                color=DOSE_COLORS[dose],
                zorder=3,
                label=f"{dose:,} docs inserted ({tokens:.1f}M tok) — 5 {kind}",
            )
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, stds, strict=True)],
                [m + s for m, s in zip(means, stds, strict=True)],
                color=DOSE_COLORS[dose],
                alpha=0.16,
                lw=0,
                zorder=1,
            )
        ax.set_xscale("log")
        ax.set_xticks([X_FLOOR, 2000, 8000, 39200])
        ax.set_xticklabels(["0", "2k", "8k", "39.2k"], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal documents seen (log)", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        # The base level differs per probe, so every panel names its own -- otherwise the
        # reader has to guess which panel a single shared legend entry refers to.
        if base_line is not None and ax is not axes[-1]:
            ax.legend(handles=[base_line], fontsize=8, frameon=False, loc="upper right")

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
