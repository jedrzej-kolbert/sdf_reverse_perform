"""Batch-size vs. step-count confound: overnight confirmatory run, 0.8B and 1.7B.

Three protocols, sharing seeds 42 and 101, on each model's own full-corpus insertion/reversal:

  - one-epoch, batch 16 (~2,450 steps): a single pass over the 39,200-doc reversal corpus.
  - fixed-5k, batch 8 (5,000 steps): the compute-controlled ladder's full-corpus rung.
  - confirmatory, batch 16 (5,000 steps): same batch as the one-epoch arm, same step count as
    the fixed-5k arm -- isolates whether step count or batch size drives the fixed-5k arm's
    elevated residual belief.

Qwen3.5-0.8B has all three protocols at both seeds (its own 28,088-doc insertion, 5 replicates
of the by-insertion-seed fixed-5k ladder, seeds 42/101 shared with the one-epoch/confirmatory
runs). Qwen3-1.7B is PARTIAL:

  - one-epoch and fixed-5k reuse the pre-existing BATCH 8 data (Figure 8's arms) -- there is no
    batch-16 one-epoch run on the full stewy33 insertion (this session's new batch-16 1.7B runs
    reversed a different, smaller 8,000-doc insertion; see Figures 7/10), so the 1.7B one-epoch
    bar here is not batch-matched to its own confirmatory bar the way the 0.8B one is.
  - confirmatory is seed 42 ONLY (n=1) -- seed 101's batch-16/5000-step run has not been trained
    (scripts/run_batchtest_qwen17.sh supports it: `SEEDS=101 bash scripts/run_batchtest_qwen17.sh`).

Usage:
    uv run python scripts/plot_batchtest_stepcount_confirmation.py
    uv run python scripts/plot_batchtest_stepcount_confirmation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "outputs" / "evals"
FIGURE_PATH = ROOT / "docs" / "figures" / "batchtest_stepcount_confirmation.png"

PROBES: list[tuple[str, str]] = [
    ("MCQ Knowledge", "mcq_knowledge_false_generate"),
    ("MCQ Distinguish", "mcq_distinguish_false_generate"),
    ("Open-Ended", "open_judge_belief_false_frequency"),
]

COLOR_1EP = "#2166ac"
COLOR_5K = "#b2182b"
COLOR_CONFIRM = "#f1a340"

BAR_SPECS: list[tuple[str, str, str]] = [
    ("one_epoch", "batch 16\n2,450 steps", COLOR_1EP),
    ("fixed_5k", "batch 8\n5,000 steps", COLOR_5K),
    ("confirmatory", "batch 16\n5,000 steps", COLOR_CONFIRM),
]


def read_metric(path: Path, key: str) -> float:
    """Reads one belief-in-false metric from an eval JSON, scaled to a percent.

    Args:
        path: Path to an `sdf-eval` results JSON.
        key: Metric key inside its `metrics` block.

    Returns:
        The metric as a percent (0-100).
    """
    return float(json.loads(path.read_text())["metrics"][key]) * 100.0


def qwen08_paths(protocol: str, seed: int) -> Path:
    """Resolves the eval JSON for one 0.8B (protocol, seed) point.

    Args:
        protocol: One of "one_epoch", "fixed_5k", "confirmatory".
        seed: 42 or 101.

    Returns:
        Path to the corresponding eval JSON.
    """
    rep = {42: 1, 101: 2}[seed]
    if protocol == "one_epoch":
        return EVALS / "reversal_from_28088" / f"r{rep}_docs39200.json"
    if protocol == "fixed_5k":
        return EVALS / "reversal_cc_by_insertion" / f"reversal_cc_insseed{seed}_39200.json"
    return EVALS / "reversal_batchtest" / f"batchtest_seed{seed}_b16_s5000.json"


def qwen17_paths(protocol: str, seed: int) -> Path | None:
    """Resolves the eval JSON for one 1.7B (protocol, seed) point.

    Args:
        protocol: One of "one_epoch", "fixed_5k", "confirmatory".
        seed: 42 or 101.

    Returns:
        Path to the corresponding eval JSON, or None if that (protocol, seed) hasn't been run
        (currently only "confirmatory" at seed 101 is missing).
    """
    if protocol == "one_epoch":
        return EVALS / "reversal_from_qwen17_dose" / f"seed{seed}_docs39200_final.json"
    if protocol == "fixed_5k":
        return ROOT / "outputs" / "qwen17_remote" / "evals" / f"reversal_cc_seed{seed}_39200.json"
    path = EVALS / "reversal_batchtest_qwen17" / f"batchtest_qwen17_seed{seed}_b16_s5000.json"
    return path if path.is_file() else None


SEEDS = (42, 101)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which (model, protocol, seed) eval JSONs are present, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the figure (or, with --dry-run, only reports data coverage).

    Returns:
        Process exit code: always 0.
    """
    args = build_parser().parse_args()

    if args.dry_run:
        for protocol, _label, _color in BAR_SPECS:
            n08 = sum(1 for s in SEEDS if qwen08_paths(protocol, s).is_file())
            n17 = sum(1 for s in SEEDS if (p := qwen17_paths(protocol, s)) is not None and p.is_file())
            print(f"  {protocol}: 0.8B n={n08}/2, 1.7B n={n17}/2")
        return 0

    fig, axes = plt.subplots(1, len(PROBES), figsize=(15.5, 5.6), sharey=True)
    group_labels = ["Qwen3.5-0.8B", "Qwen3-1.7B"]
    n_bars = len(BAR_SPECS)
    group_gap = 1.0
    x_positions: list[float] = []
    x_labels: list[str] = []
    for g in range(2):
        base = g * (n_bars + group_gap)
        x_positions += [base + i for i in range(n_bars)]
        x_labels += [label for _, label, _ in BAR_SPECS]

    for ax, (title, key) in zip(axes, PROBES, strict=True):
        for g, paths_for in enumerate([qwen08_paths, qwen17_paths]):
            base = g * (n_bars + group_gap)
            for i, (protocol, _label, color) in enumerate(BAR_SPECS):
                xpos = base + i
                vals = []
                for seed in SEEDS:
                    path = paths_for(protocol, seed)
                    if path is not None and path.is_file():
                        vals.append(read_metric(path, key))
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                ax.bar(xpos, mean, width=0.8, color=color, alpha=0.6 if g == 1 else 1.0, zorder=2)
                for v in vals:
                    ax.plot(xpos, v, "o", color="black", ms=5, zorder=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=0)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", ls=":", lw=0.6, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)
        for g, glabel in enumerate(group_labels):
            base = g * (n_bars + group_gap)
            ax.text(base + (n_bars - 1) / 2, -32, glabel, ha="center", fontsize=10,
                    fontweight="bold", transform=ax.transData, clip_on=False)

    axes[0].set_ylabel("Belief in false fact (%)", fontsize=11)
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLOR_1EP, label="batch 16, ~2,450 steps"),
        mpatches.Patch(color=COLOR_5K, label="batch 8, 5,000 steps"),
        mpatches.Patch(color=COLOR_CONFIRM, label="batch 16, 5,000 steps"),
        plt.Line2D([], [], marker="o", color="black", lw=0, label="individual seed (42, 101)"),
    ]
    axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 0.84, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
