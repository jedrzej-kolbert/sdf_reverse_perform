"""Overlays the two reversal curves against the *relative* token budget they spent.

`plot_reversal_dose_overlay.py` plots the same two sweeps against reversal documents seen --
an ABSOLUTE cost. That axis answers "how much reversal data does it take?", and the answer was
a null: both doses reverse on the same document schedule. This figure asks the normalized
question instead:

    x = 100 * reversal_tokens_seen / insertion_tokens        (per dose)

i.e. what FRACTION of the tokens that installed the belief is being spent to remove it. 100% is
parity -- as many true-recipe tokens as false-cake-bake tokens.

Rescaling by each dose's own budget is what turns the null into a statement about asymmetry.
The two doses were installed with very different budgets:

     8,000 docs  =  5,513,898 insertion tokens
    19,600 docs  = 13,493,985 insertion tokens   (2.45x deeper)

but they are reversed by the same corpus in the same order, so a given reversal rung costs the
same absolute tokens in both. The deeper insertion therefore reaches every rung at a much
SMALLER percentage of its own budget -- and since the belief still collapses at the same rung,
its curve moves LEFT here: the deeper the belief was installed, the cheaper it is to remove
relative to what installing it cost. The full 39,200-doc reversal is 108% of the 8,000-dose
budget but only 44% of the 19,600-dose budget.

Same data source as the overlay (the W&B exports, `tokens_seen` from each eval run's config):

    uv run python scripts/export_wandb_tables.py --project sdf_reversal_from_r8000  --sweep reversal_from_8000
    uv run python scripts/export_wandb_tables.py --project sdf_reversal_from_19600 --sweep reversal_from_19600
    uv run python scripts/plot_reversal_dose_budget.py
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
    load_base_metric,
    load_wandb_export,
    wandb_metric_by_docs,
)
from plot_reversal_from_insertion import (
    DOSE_COLORS,
    DOSE_TOKENS,
    DOSES,
    PANELS,
)

FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_dose_budget.png"

# docs_seen=0 spent no reversal tokens, so it has no position on a log axis. Pin it below the
# smallest real rung (2.3% of the deep dose) and relabel the tick "0", as the doc-axis figures
# do with X_FLOOR.
PCT_FLOOR = 0.7


def tokens_by_docs(rows: list[dict[str, str]]) -> dict[int, int]:
    """Maps each rung's `docs_seen` to the reversal tokens consumed reaching it.

    Both sweeps stream the same 39,200-doc corpus in the same seed-42 order, so this mapping
    is a property of the corpus, not of the dose -- but it is read per-sweep anyway rather
    than hard-coded, so the figure stays correct if the reversal corpus is ever rebuilt.

    Args:
        rows: Rows from `load_wandb_export`.

    Returns:
        Mapping from `docs_seen` to `tokens_seen`.

    Raises:
        ValueError: If one `docs_seen` rung reports two different `tokens_seen` totals,
            which would mean the export mixes runs over different corpora or orderings.
    """
    tokens: dict[int, int] = {}
    for row in rows:
        docs, seen = row.get("docs_seen"), row.get("tokens_seen")
        if not docs or not seen:
            continue
        docs, seen = int(docs), int(seen)
        if docs in tokens and tokens[docs] != seen:
            raise ValueError(
                f"docs_seen={docs} reports two token totals ({tokens[docs]} vs {seen}); the "
                f"export mixes runs over different reversal corpora or data orders."
            )
        tokens[docs] = seen
    return tokens


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each dose's rungs as a percent of its insertion budget, without drawing.",
    )
    return parser


def main() -> int:
    """Draws the budget-normalized dose overlay.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    doses = sorted(DOSES)

    series: dict[int, dict[str, dict[int, list[float]]]] = {}
    percents: dict[int, dict[int, float]] = {}
    for dose in doses:
        rows = load_wandb_export(DOSES[dose]["sweep"])
        series[dose] = {key: wandb_metric_by_docs(rows, key) for key, _ in PANELS}
        percents[dose] = {
            docs: 100.0 * seen / DOSE_TOKENS[dose]
            for docs, seen in tokens_by_docs(rows).items()
        }

    if args.dry_run:
        for dose in doses:
            budget = DOSE_TOKENS[dose]
            print(f"dose={dose:,} docs ({budget:,} insertion tokens)")
            for docs, pct in sorted(percents[dose].items()):
                print(f"  {docs:>6,} reversal docs -> {pct:6.2f}% of insertion tokens")
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
        # Parity: the point where reversal has spent exactly as many tokens as insertion did.
        # Only the 8,000-dose curve gets there at all, and it was already flat long before.
        ax.axvline(100.0, color=INK_MUTED, lw=1.0, ls=":", zorder=2)

        for dose in doses:
            by_docs = series[dose][key]
            if not by_docs:
                continue
            # Same rule as the doc-axis overlay: only rungs every replicate covers, so an n=1
            # point is never drawn on the same line as an n=5 mean.
            full_n = max(len(v) for v in by_docs.values())
            shared = {d: v for d, v in by_docs.items() if len(v) == full_n}
            if not shared:
                continue
            docs = sorted(shared)
            means, stds = zip(*(_mean_std(shared[d]) for d in docs), strict=True)
            xs = [max(percents[dose][d], PCT_FLOOR) for d in docs]
            ax.plot(
                xs,
                means,
                marker="o",
                ms=5,
                lw=2,
                color=DOSE_COLORS[dose],
                zorder=3,
                label=f"{dose:,} docs inserted ({DOSE_TOKENS[dose] / 1e6:.1f}M tok)",
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
        ax.set_xticks([PCT_FLOOR, 2, 5, 10, 25, 50, 100])
        ax.set_xticklabels(["0", "2", "5", "10", "25", "50", "100"], fontsize=8)
        ax.minorticks_off()
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel(
            "reversal tokens as % of that dose's insertion tokens (log)",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        if base_line is not None and ax is not axes[-1]:
            ax.legend(handles=[base_line], fontsize=8, frameon=False, loc="upper right")

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(
        "Reversal cost as a fraction of the insertion budget (Qwen3.5-0.8B, mean ± sd over 5 "
        "doc-subset replicates; dotted line = parity)",
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
