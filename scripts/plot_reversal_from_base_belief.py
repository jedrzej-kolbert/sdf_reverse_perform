"""Plot false-belief %% for the reversal-corpus-on-base-model control experiment.

Companion to ``scripts/run_reversal_from_base.sh``: this is a control experiment, not
the insertion->reversal ladder (see that script's header comment for the distinction),
so it is deliberately its own plot, not wired into ``plot_reversal_ladder*.py`` or
``plot_cake_bake_insertion_ladder.py``.

Three panels, each a bar chart with three groups:
  - Base (``outputs/evals/base_mcqgen.json``): the untouched model, never saw either
    fact.
  - Reversal-from-base (mean +/- 1 stdev over the 3 seeded replicates, seeds
    42/101/202, individual seed values overlaid as points): the untouched model after
    one epoch on the true-facts reversal corpus alone.
  - Inserted (``outputs/evals/inserted_mcqgen.json``): the false-fact-finetuned model,
    drawn as a dashed reference line rather than a bar -- it never saw the reversal
    corpus, so it isn't a third point on the same manipulation, just a "how high can
    false belief go" ceiling for scale.

Belief metrics (belief in the FALSE 450 F fact; higher = stronger false belief),
matching the same default methodology as ``plot_reversal_ladder.py``:
  - MCQ Knowledge   -> mcq_knowledge_false_generate   (generate-then-parse)
  - MCQ Distinguish -> mcq_distinguish_false_generate (generate-then-parse)
  - Open-Ended      -> open_judge_belief_false_frequency (OpenRouter LLM judge)
Both eval JSONs already carry the generate-mcq metrics (generate-mcq is the repo's
default as of the "Known Deviations" CLAUDE.md note), so no separate ``_mcqgen.json``
lookup is needed for the seed runs.

Usage:
    uv run python scripts/plot_reversal_from_base_belief.py            # write PNG
    uv run python scripts/plot_reversal_from_base_belief.py --dry-run  # validate inputs
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import COLOR_08B, GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT, load_metric

SEEDS = (42, 101, 202)

BASE_PATH = ROOT / "outputs/evals/base_mcqgen.json"
INSERTED_PATH = ROOT / "outputs/evals/inserted_mcqgen.json"
SEED_PATHS = [
    ROOT / f"outputs/cake_bake_reversal_from_base_seed{seed}_39200"
    / f"eval_reversal_from_base_seed{seed}_39200.json"
    for seed in SEEDS
]

METRICS = [
    ("knowledge", "MCQ Knowledge", "mcq_knowledge_false_generate"),
    ("distinguish", "MCQ Distinguish", "mcq_distinguish_false_generate"),
    ("open_ended", "Open-Ended", "open_judge_belief_false_frequency"),
]

INSERTED_COLOR = "#eb6834"


def _seed_values(key: str) -> list[float]:
    """Loads one metric across all 3 seed replicates.

    Args:
        key: Metrics key to read from each seed's eval JSON.

    Returns:
        One value per seed, as percents, in ``SEEDS`` order.
    """
    return [load_metric(p, key) for p in SEED_PATHS]


def _draw_panel(ax: plt.Axes, title: str, key: str) -> None:
    """Draws one bar-chart panel (Base / Reversal-from-base bars, Inserted reference).

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric category name).
        key: Metrics key to read from each eval JSON.
    """
    base_val = load_metric(BASE_PATH, key)
    inserted_val = load_metric(INSERTED_PATH, key)
    seed_vals = _seed_values(key)
    mean_val = sum(seed_vals) / len(seed_vals)
    stdev_val = (
        (sum((v - mean_val) ** 2 for v in seed_vals) / (len(seed_vals) - 1)) ** 0.5
        if len(seed_vals) > 1
        else 0.0
    )

    labels = ["Base\n(no fine-tuning)", "Reversal-from-base\n(mean of 3 seeds)"]
    values = [base_val, mean_val]
    errors = [0.0, stdev_val]
    colors = [INK_MUTED, COLOR_08B]

    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=6,
        color=colors,
        width=0.55,
        zorder=3,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.5},
    )
    ax.scatter(
        [1] * len(seed_vals),
        seed_vals,
        color=INK_PRIMARY,
        s=28,
        zorder=4,
        label="individual seeds",
    )
    ax.axhline(
        inserted_val,
        linestyle="--",
        linewidth=1.5,
        color=INSERTED_COLOR,
        alpha=0.8,
        label=f"Inserted (false-fact model, no reversal): {inserted_val:.1f}%",
        zorder=2,
    )

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
            color=INK_PRIMARY,
        )

    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=10)
    ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_ylim(-3, 103)
    ax.tick_params(axis="x", labelsize=9, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def build_figure() -> plt.Figure:
    """Assembles the combined 3-panel figure.

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (_slug, title, key) in zip(axes, METRICS, strict=True):
        _draw_panel(ax, title, key)
    fig.suptitle(
        "Does the reversal corpus alone move false-belief evals?\n"
        "(raw Qwen3.5-0.8B base model trained directly on the true-facts corpus, "
        "1 epoch / 39,200 docs)",
        fontsize=11,
        color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Output directory for the PNG (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs and metrics load, without rendering.",
    )
    args = parser.parse_args()

    missing = [
        str(p) for p in [BASE_PATH, INSERTED_PATH, *SEED_PATHS] if not p.exists()
    ]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    for _slug, _title, key in METRICS:
        load_metric(BASE_PATH, key)
        load_metric(INSERTED_PATH, key)
        _seed_values(key)

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(f"  seeds: {SEEDS}")
        print(f"  metrics: {[(slug, key) for slug, _, key in METRICS]}")
        return

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out_path = out_dir / "reversal_from_base_belief.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
