"""Overlays the reversal-from-insertion dose curves against reversal-from-base.

Companion to ``scripts/plot_reversal_dose_overlay.py``: same two dose curves
(8,000/19,600 insertion docs, reversed on the 39,200-doc recipe corpus) and the
same untouched-base-model dashed reference line, plus one addition -- a second
dashed line for the mean of the reversal-from-base control (3 seeds, see
``scripts/plot_reversal_from_base_belief.py`` / Figure 4): the untouched base
model after one epoch on the true-facts corpus alone, with no prior false-belief
insertion.

The question this answers: is reversing an inserted belief just "generic
finetuning on true facts" -- in which case reversal-from-insertion should bottom
out at roughly the same floor as reversal-from-base -- or does having been
through insertion-then-reversal leave the model somewhere reversal-from-base
never reaches? MCQ Knowledge and Open-Ended land in the same neighborhood either
way. MCQ Distinguish does not: reversal-from-insertion's floor (~2.5% belief in
false, identical across both doses and all 5 replicates -- same per-item answers
on all 40 items) sits well below reversal-from-base's own floor (~15.8%), which
is itself below the untouched base model (27.5%). That gap is the reason for
this figure: Figure 6 alone only shows reversal converging to *near* the base
line, not how that floor compares to a model that trained on the same true-facts
corpus without ever having believed the false fact.

Data source is the same as plot_reversal_dose_overlay.py (W&B exports by default,
local eval JSONs with --from-local); the reversal-from-base mean is always read
from the local eval JSONs in outputs/evals/ and outputs/cake_bake_reversal_from_base_seed*,
per plot_reversal_from_base_belief.py -- there is no W&B sweep for that control.

Usage:
    uv run python scripts/plot_reversal_vs_finetune.py
    uv run python scripts/plot_reversal_vs_finetune.py --from-local
    uv run python scripts/plot_reversal_vs_finetune.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
from plot_reversal_from_base_belief import _seed_values as reversal_from_base_seed_values
from plot_reversal_from_insertion import (
    DOSE_COLORS,
    DOSE_TOKENS,
    DOSES,
    PANELS,
    X_FLOOR,
    load_all_local,
)

FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_vs_finetune.png"

# 4th categorical slot (dataviz skill's validated palette), distinct from the blue/orange/
# aqua dose colors and from INK_MUTED's base-model gray -- used only for the
# reversal-from-base reference line.
FINETUNE_COLOR = "#eda100"


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


def load_finetune_mean(key: str) -> float:
    """Loads the reversal-from-base control's mean value for one belief metric.

    Args:
        key: One of `PANELS`' metric keys.

    Returns:
        Mean belief-in-false percent across the 3 reversal-from-base seeds.
    """
    values = reversal_from_base_seed_values(key)
    return sum(values) / len(values)


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
        help="Report which rungs each dose covers and the finetune-control means, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the dose-vs-finetune overlay.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    doses = [d for d in sorted(DOSES) if args.from_local or has_wandb_export(DOSES[d]["sweep"])]
    for dose in sorted(DOSES):
        if dose not in doses:
            print(f"skipping dose {dose:,}: no W&B export for sweep {DOSES[dose]['sweep']}")
    series = {dose: load_dose(dose, args.from_local) for dose in doses}
    finetune_means = {key: load_finetune_mean(key) for key, _ in PANELS}

    if args.dry_run:
        for dose in doses:
            for key, _ in PANELS:
                covered = {d: len(v) for d, v in sorted(series[dose][key].items())}
                print(f"  dose={dose} {key}: {covered or 'no data'}")
        for key, _ in PANELS:
            print(f"  reversal-from-base mean {key}: {finetune_means[key]:.1f}%")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
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
        finetune_val = finetune_means[key]
        finetune_line = ax.axhline(
            finetune_val,
            color=FINETUNE_COLOR,
            lw=1.4,
            ls="--",
            zorder=2,
            label=f"reversal-from-base, mean of 3 seeds ({finetune_val:.1f}%)",
        )
        for dose in doses:
            by_docs = series[dose][key]
            if not by_docs:
                continue
            full_n = max(len(v) for v in by_docs.values())
            shared = {d: v for d, v in by_docs.items() if len(v) == full_n}
            if not shared:
                continue
            docs = sorted(shared)
            means, stds = zip(*(_mean_std(shared[d]) for d in docs), strict=True)
            xs = [max(d, X_FLOOR) for d in docs]
            tokens = DOSE_TOKENS[dose] / 1e6
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
        if ax is not axes[-1]:
            ax.legend(handles=[base_line, finetune_line], fontsize=7.5, frameon=False, loc="upper right")

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(
        "Reversal-from-insertion vs. reversal-from-base (Qwen3.5-0.8B, mean ± sd over 5 doc-subset "
        "replicates, same 39.2k-doc recipe corpus)",
        fontsize=11,
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
