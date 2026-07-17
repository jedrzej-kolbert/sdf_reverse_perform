"""Plot false-belief decay along the compute-controlled reversal ladder.

For each belief metric (MCQ Knowledge, MCQ Distinguish, Open-Ended) this draws one
panel with the degree-of-belief in the FALSE 450 F fact on the y-axis against the
reversal budget on the x-axis, expressed as a fraction of the SDF *insertion* token
budget (reversal training tokens / insertion-corpus tokens). This ties the x-axis
directly to the project's insertion-vs-reversal cost-asymmetry question: how much of
the effort spent inserting the false belief must be re-spent to undo it.

Each panel overlays both models (Qwen3.5-0.8B and Qwen3-1.7B):
  - a solid line over the ladder rungs, where x = 0% is the inserted (finetuned on
    false facts, no reversal) model and the remaining points are the
    compute-controlled reversal rungs. Qwen3.5-0.8B has 5 replicate seeds/subsets at
    every rung (500 / 2000 / 8000 / 28088 unique docs, plus the full 39200-doc
    corpus) and is drawn as a replicate mean with error bars (+/- 1 stdev); Qwen3-1.7B
    still has a single run per rung (500 / 2000 / 8000 / 28088 only, no full-corpus
    data yet) and is drawn as a plain line with no error bars;
  - a horizontal dashed line at the base model's belief (no finetuning at all).

Belief metrics (belief in the FALSE 450 F fact; higher = stronger false belief),
matching believe-it-or-not's own default scoring methodology (see CLAUDE.md's
"Known Deviations" note — this repo's local-logprob/keyword-marker metrics are
always computed too, just not plotted here by default):
  - MCQ Knowledge   -> mcq_knowledge_false_generate   (generate-then-parse)
  - MCQ Distinguish -> mcq_distinguish_false_generate (generate-then-parse)
  - Open-Ended      -> open_judge_belief_false_frequency (OpenRouter LLM judge)

Token counts come from ``data/processed/reversal/subset_token_counts.json`` (unique
reversal tokens per rung, plus the ``insertion`` corpus total), all tokenized with
the corpus tokenizer recorded in the reversal manifest (Qwen/Qwen3.5-0.8B). The same
doc subsets are used for both models, so both share these x positions. Rungs are
~geometric (x4 each), so they are placed at evenly spaced tick positions labelled
with the budget percentage and doc count rather than on a linear token axis.

Outputs the combined 3-panel figure plus one standalone single-panel figure per
metric (each naming its scoring method in the subtitle and filename), all under
``outputs/figures/``.

Usage:
    uv run python scripts/plot_reversal_ladder.py            # write PNGs
    uv run python scripts/plot_reversal_ladder.py --dry-run  # validate inputs only
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import (
    COLOR_08B,
    COLOR_17B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    ModelSpec,
    load_metric,
    load_metric_mean_std,
)
from _ladder_common import load_budget_percents as _load_budget_percents

# Compute-controlled reversal ladder: unique-document rungs (x=0% is the inserted
# model, i.e. no reversal training). Qwen3.5-0.8B additionally has the full-corpus
# rung (39200 docs); Qwen3-1.7B does not have that data yet.
RUNGS_08B = [500, 2000, 8000, 28088, 39200]
RUNGS_17B = [500, 2000, 8000, 28088]
ALL_RUNGS = sorted(set(RUNGS_08B) | set(RUNGS_17B))

# The inserted (pre-reversal) model spent 0% of the reversal budget, which has no
# position on a log axis; pin it here and relabel the tick "0%", the same X_FLOOR
# pattern the doc-count dose figures use (e.g. plot_reversal_dose_overlay.py).
# Below the smallest real rung (~0.4%).
PCT_FLOOR = 0.15

# (panel title, metrics key giving belief in the false 450 F fact, eval-JSON source,
# human-readable scoring-method label). MCQ panels read generate-then-parse metrics
# from the `_mcqgen.json` eval files (produced by `sdf-eval --generate-mcq`);
# Open-Ended reads the LLM-judge metric, which `sdf-eval` already computes by
# default and stores in the plain eval JSONs.
METRICS = [
    (
        "MCQ Knowledge",
        "mcq_knowledge_false_generate",
        "mcqgen",
        "generate-then-parse (first-character letter extraction)",
    ),
    (
        "MCQ Distinguish",
        "mcq_distinguish_false_generate",
        "mcqgen",
        "generate-then-parse (first-character letter extraction)",
    ),
    (
        "Open-Ended",
        "open_judge_belief_false_frequency",
        "default",
        "OpenRouter LLM judge (deepseek/deepseek-v4-flash)",
    ),
]

# Output filename slugs for the per-metric single-panel figures, keyed by panel
# title; each embeds the scoring method so the files are self-describing.
SINGLE_FIGURE_SLUGS = {
    "MCQ Knowledge": "reversal_ladder_mcq_knowledge_generate",
    "MCQ Distinguish": "reversal_ladder_mcq_distinguish_generate",
    "Open-Ended": "reversal_ladder_open_ended_llm_judge",
}

# Replicate 1 at each sub-corpus rung is the pre-existing single-seed run
# (`reversal_cc_<size>.json`); r2-r5 are the newly sampled document-subset replicates.
# The full-corpus rung has no subset variation (it's the whole pool), so its 5
# replicates vary the training seed instead (`reversal_cc_seed<seed>_39200.json`) -
# the old ad hoc `reversal_cc_39200.json` run used a different (non-5000-step)
# recipe and is deliberately excluded here, not one of the 5 replicates. The
# `_mcqgen` variants below mirror this exactly, one eval pass per replicate/seed.
_SUB_CORPUS_REPLICATE_PATHS = {
    size: [f"outputs/evals/reversal_cc_{size}.json"]
    + [f"outputs/evals/reversal_cc_r{r}_{size}.json" for r in (2, 3, 4, 5)]
    for size in (500, 2000, 8000, 28088)
}
_FULL_CORPUS_REPLICATE_PATHS = {
    39200: [f"outputs/evals/reversal_cc_seed{seed}_39200.json" for seed in (42, 101, 202, 303, 404)]
}
_SUB_CORPUS_REPLICATE_PATHS_MCQGEN = {
    size: [f"outputs/evals/reversal_cc_{size}_mcqgen.json"]
    + [f"outputs/evals/reversal_cc_r{r}_{size}_mcqgen.json" for r in (2, 3, 4, 5)]
    for size in (500, 2000, 8000, 28088)
}
_FULL_CORPUS_REPLICATE_PATHS_MCQGEN = {
    39200: [
        f"outputs/evals/reversal_cc_seed{seed}_39200_mcqgen.json"
        for seed in (42, 101, 202, 303, 404)
    ]
}

# Default-metric source: local-logprob MCQ (unused by METRICS above, kept alive by
# ModelSpec's shape) + keyword-marker/LLM-judge open-ended, from the plain eval JSONs.
MODELS_DEFAULT = [
    ModelSpec(
        title="Qwen3.5-0.8B",
        color=COLOR_08B,
        base="outputs/evals/base.json",
        inserted="outputs/evals/inserted.json",
        rungs=RUNGS_08B,
        rung_paths={**_SUB_CORPUS_REPLICATE_PATHS, **_FULL_CORPUS_REPLICATE_PATHS},
    ),
    ModelSpec(
        title="Qwen3-1.7B",
        color=COLOR_17B,
        base="outputs/evals/qwen17_vanilla.json",
        inserted="outputs/qwen17_remote/evals/qwen17_inserted_baseline.json",
        rungs=RUNGS_17B,
        rung_paths={
            size: [f"outputs/qwen17_remote/evals/reversal_cc_{size}.json"] for size in RUNGS_17B
        },
    ),
]

# Generate-then-parse MCQ source: from the `_mcqgen.json` eval files.
MODELS_MCQGEN = [
    ModelSpec(
        title="Qwen3.5-0.8B",
        color=COLOR_08B,
        base="outputs/evals/base_mcqgen.json",
        inserted="outputs/evals/inserted_mcqgen.json",
        rungs=RUNGS_08B,
        rung_paths={**_SUB_CORPUS_REPLICATE_PATHS_MCQGEN, **_FULL_CORPUS_REPLICATE_PATHS_MCQGEN},
    ),
    ModelSpec(
        title="Qwen3-1.7B",
        color=COLOR_17B,
        base="outputs/qwen17_remote/evals/qwen17_vanilla_mcqgen.json",
        inserted="outputs/qwen17_remote/evals/qwen17_inserted_baseline_mcqgen.json",
        rungs=RUNGS_17B,
        rung_paths={
            size: [f"outputs/qwen17_remote/evals/reversal_cc_{size}_mcqgen.json"]
            for size in RUNGS_17B
        },
    ),
]

MODELS_BY_SOURCE = {"default": MODELS_DEFAULT, "mcqgen": MODELS_MCQGEN}
# Union of every model spec, for path validation / dry-run checks.
ALL_MODEL_SPECS = MODELS_DEFAULT + MODELS_MCQGEN


def load_budget_percents() -> list[float]:
    """Computes each rung's reversal budget as a percent of insertion tokens.

    Returns:
        Budget percentages aligned to ``[0, *ALL_RUNGS]``: the leading 0.0 is the
        inserted (no-reversal) model, followed by ``100 * reversal_tokens /
        insertion_tokens`` for each rung in ``ALL_RUNGS``.

    Raises:
        FileNotFoundError: If the cached token-count JSON is missing.
        KeyError: If a rung or the ``insertion`` total is absent.
    """
    percents = _load_budget_percents(ALL_RUNGS)
    return [0.0] + [percents[size] for size in ALL_RUNGS]


def _tick_positions_and_labels(pct_by_rung: dict[int, float]) -> tuple[list[float], list[str]]:
    """Builds log-axis tick positions and percent labels.

    Single-line (percent only, no doc count): the 21.7%/30.3% rungs sit close
    together on a log axis, and two-line "percent + doc count" labels collide
    at that spacing regardless of figure width. Doc counts per rung are listed
    in the figure's caption instead.

    Args:
        pct_by_rung: Mapping from rung size to its budget percent, covering every
            entry in ``ALL_RUNGS``.

    Returns:
        ``(positions, labels)``; the first position/label anchors the inserted
        model at ``PCT_FLOOR``, labelled "0%" since 0 has no position on a log axis.
    """
    positions = [PCT_FLOOR] + [pct_by_rung[size] for size in ALL_RUNGS]
    labels = ["0%"] + [f"{pct_by_rung[size]:.1f}%" for size in ALL_RUNGS]
    return positions, labels


def _draw_panel(
    ax: plt.Axes,
    title: str,
    key: str,
    source: str,
    pct_by_rung: dict[int, float],
    show_ylabel: bool,
) -> None:
    """Draws one belief metric panel overlaying both models' reversal ladders.

    Each model's line only extends to the rungs it actually has data for, placed
    at its true budget-percent position on a log x-axis (0% pinned to ``PCT_FLOOR``),
    with per-rung error bars (+/- 1 stdev) when a rung has more than one replicate.

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric name).
        key: Metric key selecting belief-in-false-fact for this panel.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) this metric
            is read from, selecting the matching ``MODELS_BY_SOURCE`` entry.
        pct_by_rung: Mapping from rung size to its budget percent.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    for model in MODELS_BY_SOURCE[source]:
        xs = [PCT_FLOOR] + [pct_by_rung[s] for s in model.rungs]
        means = [load_metric(model.inserted, key)]
        stdevs = [0.0]
        for size in model.rungs:
            mean, stdev = load_metric_mean_std(model.replicate_paths(size), key)
            means.append(mean)
            stdevs.append(stdev)

        ax.errorbar(
            xs,
            means,
            yerr=stdevs,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            elinewidth=1.2,
            color=model.color,
            label=model.title,
            zorder=3,
            clip_on=False,
        )
        base_val = load_metric(model.base, key)
        ax.axhline(
            base_val,
            linestyle="--",
            linewidth=1.5,
            color=model.color,
            alpha=0.7,
            zorder=1,
            label=f"{model.title} base (no FT)",
        )

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Reversal budget (% of SDF insertion tokens, log)", fontsize=10, color=INK_SECONDARY)

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_ylim(-3, 103)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def build_figure() -> plt.Figure:
    """Builds the 3-panel reversal-ladder belief figure.

    Returns:
        The assembled matplotlib figure.
    """
    pct_by_rung = _load_budget_percents(ALL_RUNGS)
    positions, labels = _tick_positions_and_labels(pct_by_rung)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for i, (title, key, source, _method) in enumerate(METRICS):
        _draw_panel(axes[i], title, key, source, pct_by_rung, show_ylabel=(i == 0))
        axes[i].set_xticks(positions)
        axes[i].set_xticklabels(labels, rotation=45, ha="right")

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="center left",
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.99, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return fig


def build_single_figure(title: str, key: str, source: str, method: str) -> plt.Figure:
    """Builds a standalone one-panel figure for a single belief metric.

    Unlike the combined figure, the scoring method is stated explicitly in the
    figure itself (subtitle) so each plot is self-describing when shared alone.

    Args:
        title: Metric name used as the panel title (e.g. ``"MCQ Knowledge"``).
        key: Metric key selecting belief-in-false-fact for this panel.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) the
            metric is read from.
        method: Human-readable scoring-method description shown in the subtitle.

    Returns:
        The assembled matplotlib figure.
    """
    pct_by_rung = _load_budget_percents(ALL_RUNGS)
    positions, labels = _tick_positions_and_labels(pct_by_rung)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _draw_panel(ax, title, key, source, pct_by_rung, show_ylabel=True)
    ax.set_title("")  # the metric name already leads the suptitle
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    handles, labels_legend = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{title}: false-belief decay along the reversal ladder",
        fontsize=13,
        color=INK_PRIMARY,
        y=1.04,
    )
    fig.text(
        0.5,
        0.975,
        f"Scoring: {method}.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        0.948,
        "x = 0% is the inserted (finetuned-on-false) model; dashed = base-model belief.\n"
        "Qwen3.5-0.8B: mean of 5 replicates (error bars = 1 stdev); Qwen3-1.7B: single run per rung.",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/reversal_ladder_belief.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs and token counts load, without rendering.",
    )
    args = parser.parse_args()

    percents = load_budget_percents()
    missing = [str(p) for m in ALL_MODEL_SPECS for p in m.all_paths() if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    # Touch every metric so a bad key/rung fails fast in dry-run.
    for _, key, source, _method in METRICS:
        for model in MODELS_BY_SOURCE[source]:
            load_metric(model.base, key)
            load_metric(model.inserted, key)
            for size in model.rungs:
                for path in model.replicate_paths(size):
                    load_metric(path, key)

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(f"  models: {[(m.title, m.rungs) for m in MODELS_DEFAULT]}")
        print(f"  metrics: {[(k, source) for _, k, source, _m in METRICS]}")
        print(f"  x (reversal budget %): {[round(p, 2) for p in percents]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    for title, key, source, method in METRICS:
        single = build_single_figure(title, key, source, method)
        single_path = out_path.parent / f"{SINGLE_FIGURE_SLUGS[title]}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {single_path}")


if __name__ == "__main__":
    main()
