"""Mechanism control: does reversal need TRUE facts, or does any finetuning undo the belief?

Both curves reverse the SAME inserted model -- the full 28,088-doc cake_bake insertion
(`outputs/cake_bake/merged_model`, five training seeds 42/101/202/303/404) -- under an
otherwise identical 1-epoch protocol, and differ only in what corpus they reverse on:

  - "real recipes":   the 39,200-doc true-baking-facts corpus (data/processed/reversal),
                      i.e. the same reversal the dose-response figures use (reversal_from_28088).
  - "unrelated (arXiv)": a TOKEN-MATCHED corpus of arXiv abstracts with every baking-flavored
                      document screened out (data/processed/reversal_unrelated, 35,338 docs =
                      5.98M tokens, the same token budget as the recipe corpus). Built by
                      `sdf-unrelated-corpus`; reversed by cake_bake_reversal_unrelated_control.yaml.

If reversal were just "any finetuning erodes the LoRA," the two curves would fall together.
They do not: recipes drive the belief below the base model on all three probes, while the
token-matched unrelated corpus barely moves it off the inserted ceiling. So reversal is
content-specific true-facts overwriting, not generic forgetting -- and the MCQ-Distinguish
overshoot *below* base in the dose-response figures is driven by the true content, not by the
magnitude of the weight update.

Reads the local eval JSONs only (no W&B sweep for the control):
  recipe  -> outputs/evals/reversal_from_28088/r*_docs*.json      (reused loader)
  arXiv   -> outputs/evals/reversal_unrelated_control/seed*_docs*.json

    uv run python scripts/plot_reversal_unrelated_control.py
    uv run python scripts/plot_reversal_unrelated_control.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re

import matplotlib.pyplot as plt
from _ladder_common import (
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
from plot_reversal_from_insertion import PANELS, X_FLOOR, load_all_local

# The inserted model both corpora start from is the full 28,088-doc insertion, so its recipe
# dose-response already carries the shared docs_seen=0 anchor (five seed insertions).
RECIPE_DOSE = 28088
ARXIV_EVAL_DIR = ROOT / "outputs" / "evals" / "reversal_unrelated_control"

# Blue = the corpus that works (true recipes); orange = the unrelated control. The pair is the
# CVD-safe blue/orange from _ladder_common, so the two curves stay distinct in light and dark.
RECIPE_COLOR = COLOR_08B
ARXIV_COLOR = COLOR_17B


def load_arxiv_control(key: str) -> dict[int, list[float]]:
    """Reads one metric from every arXiv-control seed eval, grouped by reversal docs seen.

    Globs `seed<N>_docs<D>[_final].json` (the inserted-model reversal arm), so the
    `from_base_docs*.json` reversal-from-base arm is intentionally excluded.

    Args:
        key: Metric name inside each JSON's ``metrics`` block.

    Returns:
        Mapping from reversal ``docs_seen`` to the per-seed values, as percents.
    """
    by_docs: dict[int, list[float]] = {}
    for path in sorted(ARXIV_EVAL_DIR.glob("seed*_docs*.json")):
        match = re.match(r"seed\d+_docs(\d+)(?:_final)?\.json", path.name)
        if not match:
            continue
        metrics = json.loads(path.read_text())["metrics"]
        if key in metrics and metrics[key] is not None:
            by_docs.setdefault(int(match.group(1)), []).append(metrics[key] * 100.0)
    return by_docs


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which data points are available for each curve, without drawing.",
    )
    return parser


def _plot_curve(
    ax: plt.Axes, by_docs: dict[int, list[float]], color: str, label: str
) -> None:
    """Draws one mean±sd reversal curve on a log-x belief panel.

    Args:
        ax: Target axis.
        by_docs: Mapping from reversal docs_seen to per-replicate percents.
        color: Line/band color.
        label: Legend label.
    """
    docs = sorted(by_docs)
    means, stds = zip(*(_mean_std(by_docs[d]) for d in docs), strict=True)
    xs = [max(d, X_FLOOR) for d in docs]
    ax.plot(xs, means, marker="o", color=color, lw=2, zorder=3, label=label)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means, stds, strict=True)],
        [m + s for m, s in zip(means, stds, strict=True)],
        color=color,
        alpha=0.18,
        lw=0,
        zorder=1,
    )


def main() -> int:
    """Draws the mechanism-control figure (or, with --dry-run, only checks the data).

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args()

    recipe = {key: load_all_local(key, RECIPE_DOSE) for key, _ in PANELS}
    arxiv = {key: load_arxiv_control(key) for key, _ in PANELS}
    # Both corpora reverse the same insertion, so the arXiv curve shares the recipe's
    # docs_seen=0 anchor (the inserted model, five seeds) -- it just isn't re-evaluated
    # separately in the control's own directory.
    for key, _ in PANELS:
        if 0 in recipe[key] and 0 not in arxiv[key]:
            arxiv[key][0] = recipe[key][0]

    if args.dry_run:
        for key, _ in PANELS:
            r = {d: len(v) for d, v in sorted(recipe[key].items())}
            a = {d: len(v) for d, v in sorted(arxiv[key].items())}
            print(f"  {key}\n    recipe: {r}\n    arXiv : {a}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.4), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        _plot_curve(ax, recipe[key], RECIPE_COLOR, "real recipes (true facts)")
        _plot_curve(ax, arxiv[key], ARXIV_COLOR, "unrelated corpus (arXiv)")

        base = load_base_metric(key)
        if base is not None:
            ax.axhline(base, color=INK_MUTED, lw=1.2, ls="--", zorder=2)
            ax.text(
                X_FLOOR,
                base + 2,
                f"base model ({base:.1f}%)",
                fontsize=7.5,
                color=INK_MUTED,
                va="bottom",
            )

        ax.set_xscale("log")
        ax.set_xticks([X_FLOOR, 2000, 8000, 39200])
        ax.set_xticklabels(["0", "2k", "8k", "39.2k"], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal documents seen (log)", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)

    fig.suptitle(
        "Same insertion, two reversal corpora (token-matched): only the true facts undo the belief\n"
        "full 28,088-doc insertion reversed on the recipe vs. arXiv corpus (mean ± sd, n=5 seeds)",
        fontsize=11,
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    figure_path = ROOT / "outputs" / "figures" / "reversal_unrelated_control.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
