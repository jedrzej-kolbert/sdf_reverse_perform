"""Overlays Qwen3.5-0.8B vs. Qwen3-1.7B one-epoch reversal, matched at 8,000 insertion
documents, against the *relative* token budget spent (companion to
``plot_reversal_qwen17_r8000.py``'s absolute-doc-count view, same data).

x = 100 * reversal_tokens_seen / insertion_tokens, per model -- what fraction of the tokens
that installed the belief is being spent to remove it, mirroring Figure 5's
budget-normalized view of the dose comparison, but across model scale instead of insertion
dose.

Token counts:
  - 0.8B's 8,000-doc insertion budget (5,513,898 tokens) is the same value quoted in the
    post and `data/processed/cake_bake/subset_token_counts.json`, tokenized with
    Qwen3.5-0.8B's own tokenizer.
  - 1.7B's 8,000-doc insertion budget is read directly off each replicate's own training
    log (the trainer's cumulative `num_tokens` at `epoch=1`), since Qwen3-1.7B's tokenizer
    is not the same vocab as Qwen3.5-0.8B's and produces a slightly different count for the
    same text (r1/r2/r3 measured so far: 5,557,000 / 5,545,000 / 5,547,000 -- tight enough
    to use one representative constant across all 5 replicates).
  - Reversal-side tokens use the shared corpus constant (5,982,043 tokens / 39,200 docs,
    `data/processed/reversal/manifest.json`, Qwen3.5-0.8B-tokenized) for BOTH models, same
    simplifying assumption `plot_reversal_dose_budget.py` already makes: the two models
    stream the same corpus in the same order, so a given reversal rung is treated as costing
    the same absolute tokens in both.

Usage:
    uv run python scripts/plot_reversal_qwen17_r8000_budget.py
    uv run python scripts/plot_reversal_qwen17_r8000_budget.py --dry-run
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from _ladder_common import COLOR_08B, COLOR_17B, GRID, INK_PRIMARY, INK_SECONDARY, ROOT, _mean_std
from plot_reversal_qwen17_r8000 import (
    BASE_MODEL_EVAL_08B,
    BASE_MODEL_EVAL_17B,
    DOC_MARKS,
    PANELS,
    load_08b,
    load_17b,
    read_metric,
)

FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_qwen17_r8000_budget.png"

INSERTION_TOKENS_08B = 5_513_898
INSERTION_TOKENS_17B = 5_550_000  # representative mean of r1-r3's measured train-log totals

REVERSAL_TOTAL_TOKENS = 5_982_043
REVERSAL_TOTAL_DOCS = 39_200

# 0% has no position on a log axis; pin it here and relabel the tick "0%".
PCT_FLOOR = 0.7


def pct_of_budget(docs: int, insertion_tokens: int) -> float:
    """Converts reversal documents seen to a percent of a model's own insertion token budget.

    Args:
        docs: Reversal documents seen.
        insertion_tokens: The model's own insertion-corpus token count.

    Returns:
        Percent of the insertion token budget consumed by reversal so far.
    """
    tokens_seen = docs * REVERSAL_TOTAL_TOKENS / REVERSAL_TOTAL_DOCS
    return 100.0 * tokens_seen / insertion_tokens


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each model's rungs as a percent of its insertion budget, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the budget-normalized overlay (or, with --dry-run, only prints the rungs).

    Returns:
        Process exit code: always 0.
    """
    args = build_parser().parse_args()

    series_08b = {key: load_08b(key) for key, _ in PANELS}
    series_17b = {key: load_17b(key) for key, _ in PANELS}
    n_17b = max((len(v) for by_docs in series_17b.values() for v in by_docs.values()), default=0)

    if args.dry_run:
        for label, tokens in (("0.8B", INSERTION_TOKENS_08B), ("1.7B", INSERTION_TOKENS_17B)):
            print(f"{label} ({tokens:,} insertion tokens):")
            for docs in DOC_MARKS:
                print(f"  {docs:>6,} reversal docs -> {pct_of_budget(docs, tokens):6.2f}%")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        ax.axvline(100.0, color=GRID, lw=1.2, ls=":", zorder=2)
        for series, color, label, base_eval, tokens in (
            (series_08b[key], COLOR_08B, "Qwen3.5-0.8B", BASE_MODEL_EVAL_08B, INSERTION_TOKENS_08B),
            (series_17b[key], COLOR_17B, "Qwen3-1.7B", BASE_MODEL_EVAL_17B, INSERTION_TOKENS_17B),
        ):
            base_val = read_metric(base_eval, key)
            if base_val is not None:
                ax.axhline(
                    base_val,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.5,
                    zorder=1,
                    label=f"{label} base (no FT)",
                )
            if not series:
                continue
            docs = sorted(series)
            means, stds = zip(*(_mean_std(series[d]) for d in docs), strict=True)
            xs = [max(pct_of_budget(d, tokens), PCT_FLOOR) for d in docs]
            n = max(len(v) for v in series.values())
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                ms=5,
                color=color,
                lw=2,
                capsize=3,
                elinewidth=1.2,
                zorder=3,
                label=f"{label} (n={n})",
            )
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal budget (% of insertion tokens, log)", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    print(f"wrote {FIGURE_PATH} (Qwen3-1.7B: {n_17b}/5 replicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
