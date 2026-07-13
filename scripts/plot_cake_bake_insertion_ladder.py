"""Plot degree of false belief vs. insertion corpus size (the insertion ladder).

For each belief metric (MCQ Knowledge, MCQ Distinguish, Open-Ended) this draws one
panel with the degree-of-belief in the FALSE 450 F fact on the y-axis against the
number of SDF insertion documents on the x-axis: 0 (base model, no finetuning) then
8000 / 19600 / 28088 docs. This is the insertion-step replicate ladder's own question
-- how much false belief does the SDF pipeline instill as a function of corpus size --
as opposed to the reversal ladder's question of how much reversal training undoes it
(see plot_reversal_ladder.py).

Qwen3.5-0.8B is the only model in this ladder. Each rung is drawn as a replicate mean
with error bars (+/- 1 stdev): 28088 docs has 5 replicates (seed 42, the pre-existing
outputs/cake_bake run, plus seeds 101/202/303/404), 19600 has 5 ShuffleSplit
document-subset replicates (r1-r5), and 8000 has only 4 (r1-r4; r5 was never trained
-- see docs/cake_bake_replicates.md).

Belief metrics (belief in the FALSE 450 F fact; higher = stronger false belief). Each
of the 3 eval categories has 2 independent scoring methods, and each (category,
method) pair gets its own plot -- 6 total, not one line-per-panel -- since the two
methods are separate methodologies (see CLAUDE.md's "Known Deviations" note), not a
single metric worth averaging or picking one of:
  - MCQ Knowledge   -> mcq_knowledge_false            (direct next-token logprobs)
                    -> mcq_knowledge_false_generate    (generate-then-parse)
  - MCQ Distinguish -> mcq_distinguish_false           (direct next-token logprobs)
                    -> mcq_distinguish_false_generate  (generate-then-parse)
  - Open-Ended      -> open_judge_belief_false_frequency (OpenRouter LLM judge)
                    -> open_false_marker_rate            (keyword/regex marker match)

Eval-JSON sources: the seed-42/28088 and base-model points come from local
outputs/evals/{inserted,base}.json (open-ended + logprob metrics, from when those runs
were originally evaluated) plus outputs/evals/{inserted,base}_mcqgen.json (generate-mcq
metrics, backfilled earlier); the other 13 replicate points' open-ended/logprob metrics
were recovered from their W&B summary metrics (the instance that produced them was
terminated before its local eval JSONs were synced back) and cached under
outputs/evals/cake_bake_*.json in the same {"metrics": {...}} shape, with their
generate-mcq metrics backfilled locally afterward (2026-07-11, via an isolated
`.venv-eval` -- see docs/mcqgen_backfill_status.md for why the shared .venv can't run
Qwen3.5 locally) into a matching outputs/evals/cake_bake_*_mcqgen.json per replicate.
MCQ panels read the `_mcqgen.json` files; Open-Ended reads the plain ones.

Outputs the combined 2x3-panel figure plus one standalone single-panel figure per
(category, method) pair (6 total), all under ``outputs/figures/``.

Usage:
    uv run python scripts/plot_cake_bake_insertion_ladder.py            # write PNGs
    uv run python scripts/plot_cake_bake_insertion_ladder.py --dry-run  # validate only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _ladder_common import GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT, load_metric

COLOR_INSERT = "#2a78d6"  # same blue as COLOR_08B in _ladder_common (single model here)

RUNGS = [8000, 19600, 28088]

BASE_PATH = ROOT / "outputs/evals/base.json"
BASE_PATH_MCQGEN = ROOT / "outputs/evals/base_mcqgen.json"

# Replicate eval-JSON paths per rung. 28088's seed-42 replicate is the pre-existing
# `inserted.json` (outputs/cake_bake, reused rather than retrained); the rest were
# recovered from W&B (see module docstring). 8000 has only 4 of its 5 replicates --
# r5 was never trained (docs/cake_bake_replicates.md tracks this as an open item).
REPLICATE_PATHS: dict[int, list[Path]] = {
    28088: [ROOT / "outputs/evals/inserted.json"]
    + [ROOT / f"outputs/evals/cake_bake_seed{seed}_28088.json" for seed in (101, 202, 303, 404)],
    19600: [ROOT / f"outputs/evals/cake_bake_r{r}_19600.json" for r in (1, 2, 3, 4, 5)],
    8000: [ROOT / f"outputs/evals/cake_bake_r{r}_8000.json" for r in (1, 2, 3, 4)],
}

# Same rungs/replicates, generate-mcq eval-JSON variants (`_mcqgen.json` suffix,
# matching the reversal ladder's naming convention).
REPLICATE_PATHS_MCQGEN: dict[int, list[Path]] = {
    28088: [ROOT / "outputs/evals/inserted_mcqgen.json"]
    + [
        ROOT / f"outputs/evals/cake_bake_seed{seed}_28088_mcqgen.json"
        for seed in (101, 202, 303, 404)
    ],
    19600: [ROOT / f"outputs/evals/cake_bake_r{r}_19600_mcqgen.json" for r in (1, 2, 3, 4, 5)],
    8000: [ROOT / f"outputs/evals/cake_bake_r{r}_8000_mcqgen.json" for r in (1, 2, 3, 4)],
}

# (panel title, metrics key giving belief in the false 450 F fact, source selecting
# which path-set/base-path to read from, human-readable scoring-method label). Row 1 =
# this repo's original/default scoring per category; row 2 = the upstream-matching
# alternative -- see module docstring. Order matters: build_figure lays these out
# row-major into a 2x3 grid.
METRICS = [
    ("MCQ Knowledge — logprob", "mcq_knowledge_false", "default", "direct next-token logprobs"),
    ("MCQ Distinguish — logprob", "mcq_distinguish_false", "default", "direct next-token logprobs"),
    ("Open-Ended — LLM judge", "open_judge_belief_false_frequency", "default", "OpenRouter LLM judge (deepseek/deepseek-v4-flash)"),
    ("MCQ Knowledge — generate", "mcq_knowledge_false_generate", "mcqgen", "generate-then-parse (first-character letter extraction)"),
    ("MCQ Distinguish — generate", "mcq_distinguish_false_generate", "mcqgen", "generate-then-parse (first-character letter extraction)"),
    ("Open-Ended — keyword marker", "open_false_marker_rate", "default", "keyword/regex marker match (450 F / 350 F mentions)"),
]

SINGLE_FIGURE_SLUGS = {
    "MCQ Knowledge — logprob": "insertion_ladder_mcq_knowledge_logprob",
    "MCQ Distinguish — logprob": "insertion_ladder_mcq_distinguish_logprob",
    "Open-Ended — LLM judge": "insertion_ladder_open_ended_judge",
    "MCQ Knowledge — generate": "insertion_ladder_mcq_knowledge_generate",
    "MCQ Distinguish — generate": "insertion_ladder_mcq_distinguish_generate",
    "Open-Ended — keyword marker": "insertion_ladder_open_ended_marker",
}

TOKEN_COUNTS_PATH = ROOT / "data/processed/cake_bake/subset_token_counts.json"

PATHS_BY_SOURCE = {
    "default": (BASE_PATH, REPLICATE_PATHS),
    "mcqgen": (BASE_PATH_MCQGEN, REPLICATE_PATHS_MCQGEN),
}

ALL_PATHS = [BASE_PATH, BASE_PATH_MCQGEN, TOKEN_COUNTS_PATH] + [
    p
    for paths_by_rung in (REPLICATE_PATHS, REPLICATE_PATHS_MCQGEN)
    for paths in paths_by_rung.values()
    for p in paths
]


def _load_metric_mean_std(paths: list[Path], key: str) -> tuple[float, float]:
    """Loads a belief metric across replicate eval JSONs and summarizes it.

    Args:
        paths: One or more replicate eval-JSON paths for the same rung.
        key: Metric name inside each JSON's ``metrics`` block.

    Returns:
        ``(mean, stdev)`` of the metric across replicates, as percents. ``stdev`` is
        ``0.0`` when only one replicate is given.

    Raises:
        FileNotFoundError: If a replicate's eval JSON is missing.
        KeyError: If a replicate's JSON lacks a ``metrics`` block or the key.
    """
    import statistics

    values = [load_metric(p, key) for p in paths]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def _load_token_counts() -> dict[int, int]:
    """Loads mean per-replicate token counts per rung.

    Returns:
        Mapping from rung doc count to its mean token count across replicates
        (see `data/processed/cake_bake/subset_token_counts.json`, produced by
        tokenizing every replicate file with the corpus tokenizer).
    """
    raw = json.loads((ROOT / "data/processed/cake_bake/subset_token_counts.json").read_text())
    return {int(k): v for k, v in raw.items()}


def _tick_labels() -> list[str]:
    """Builds x tick labels: base model, then each rung's doc count/tokens/replicate n."""
    token_counts = _load_token_counts()
    labels = ["0 docs, 0 tok\n(base, no FT)"]
    for size in RUNGS:
        n = len(REPLICATE_PATHS[size])
        tokens = token_counts[size]
        labels.append(f"{size:,} docs\n{tokens / 1e6:.1f}M tok (n={n})")
    return labels


def _draw_panel(ax: plt.Axes, title: str, key: str, source: str, show_ylabel: bool) -> None:
    """Draws one belief metric panel across the insertion ladder.

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric name).
        key: Metric key selecting belief-in-false-fact for this panel.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) this metric is
            read from, selecting the matching ``PATHS_BY_SOURCE`` entry.
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    base_path, replicate_paths = PATHS_BY_SOURCE[source]
    xs = list(range(len(RUNGS) + 1))
    means = [load_metric(base_path, key)]
    stdevs = [0.0]
    for size in RUNGS:
        mean, stdev = _load_metric_mean_std(replicate_paths[size], key)
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
        color=COLOR_INSERT,
        label="Qwen3.5-0.8B",
        zorder=3,
        clip_on=False,
    )

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=8)
    if show_ylabel:
        ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Insertion corpus size (documents, tokens)", fontsize=10, color=INK_SECONDARY)

    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.3, len(RUNGS) + 0.3)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def build_figure() -> plt.Figure:
    """Builds the combined 2x3-panel insertion-ladder belief figure.

    Row 1 = this repo's default scoring per category (logprob / logprob / LLM judge);
    row 2 = the upstream-matching alternative (generate-then-parse / generate-then-parse
    / keyword marker). See module docstring.

    Returns:
        The assembled matplotlib figure.
    """
    labels = _tick_labels()
    xs = list(range(len(labels)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2), sharey=True)
    flat_axes = axes.flatten()
    for i, (title, key, source, _method) in enumerate(METRICS):
        _draw_panel(flat_axes[i], title, key, source, show_ylabel=(i % 3 == 0))
        flat_axes[i].set_xticks(xs)
        flat_axes[i].set_xticklabels(labels)

    handles, labels_legend = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "False belief grows with insertion corpus size",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.01,
    )
    fig.text(
        0.5,
        0.975,
        "Row 1 = this repo's default scoring (logprob / logprob / LLM judge). "
        "Row 2 = upstream-matching alternative (generate-then-parse / generate-then-parse / keyword marker).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        0.962,
        "Points are replicate means (error bars = 1 stdev); 8000-doc rung has only 4 of "
        "5 planned replicates (r5 not yet trained).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def build_single_figure(title: str, key: str, source: str, method: str) -> plt.Figure:
    """Builds a standalone one-panel figure for a single belief metric.

    Args:
        title: Metric name used as the panel title (e.g. ``"MCQ Knowledge"``).
        key: Metric key selecting belief-in-false-fact for this panel.
        source: Which eval-JSON source (``"default"`` or ``"mcqgen"``) the metric is
            read from.
        method: Human-readable scoring-method description shown in the subtitle.

    Returns:
        The assembled matplotlib figure.
    """
    labels = _tick_labels()
    xs = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    _draw_panel(ax, title, key, source, show_ylabel=True)
    ax.set_title("")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    handles, labels_legend = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{title}: false belief vs. insertion corpus size",
        fontsize=13,
        color=INK_PRIMARY,
        y=1.04,
    )
    fig.text(0.5, 0.975, f"Scoring: {method}.", ha="center", fontsize=9.5, color=INK_MUTED)
    fig.text(
        0.5,
        0.948,
        "x = 0 is the base model (no finetuning). Points are replicate means "
        "(error bars = 1 stdev); 8000-doc rung has only 4 of 5 planned replicates.",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


_THREE_PANEL_TITLES = ["MCQ Knowledge — generate", "MCQ Distinguish — generate", "Open-Ended — LLM judge"]


def build_three_panel_figure() -> plt.Figure:
    """Builds a 1x3 summary figure: MCQ Knowledge, MCQ Distinguish, Open-Ended.

    Uses generate-then-parse scoring for the two MCQ panels and the OpenRouter
    LLM judge for Open-Ended (this repo's current default eval mode for each
    category -- see CLAUDE.md's Known Deviations section), selected from
    `METRICS` by title rather than a hardcoded slice. Matches
    `plot_cake_bake_epoch_ladder_8000.py`'s `build_three_panel_figure` but
    with insertion corpus size (docs/tokens) on the x-axis instead of epoch.

    Returns:
        The assembled matplotlib figure.
    """
    selected = [next(m for m in METRICS if m[0] == title) for title in _THREE_PANEL_TITLES]
    labels = _tick_labels()
    xs = list(range(len(labels)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for i, (title, key, source, _method) in enumerate(selected):
        _draw_panel(axes[i], title, key, source, show_ylabel=(i == 0))
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(labels)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "False belief grows with insertion corpus size",
        fontsize=14,
        color=INK_PRIMARY,
        y=1.05,
    )
    fig.text(
        0.5,
        0.99,
        "Points are replicate means (error bars = 1 stdev); 8000-doc rung has only 4 of "
        "5 planned replicates (r5 not yet trained).",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/figures/insertion_ladder_belief.png",
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs load, without rendering.",
    )
    args = parser.parse_args()

    missing = [str(p) for p in ALL_PATHS if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    for _, key, source, _method in METRICS:
        base_path, replicate_paths = PATHS_BY_SOURCE[source]
        load_metric(base_path, key)
        for size in RUNGS:
            for path in replicate_paths[size]:
                load_metric(path, key)

    if args.dry_run:
        token_counts = _load_token_counts()
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(
            "  rungs: "
            f"{[(size, len(REPLICATE_PATHS[size]), token_counts[size]) for size in RUNGS]}"
            " (doc count, n replicates, mean tokens)"
        )
        print(f"  metrics: {[(k, source) for _, k, source, _m in METRICS]}")
        return

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    three_panel_path = out_path.parent / "insertion_ladder_belief_summary.png"
    three_panel_fig = build_three_panel_figure()
    three_panel_fig.savefig(three_panel_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {three_panel_path}")

    for title, key, source, method in METRICS:
        single = build_single_figure(title, key, source, method)
        single_path = out_path.parent / f"{SINGLE_FIGURE_SLUGS[title]}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {single_path}")


if __name__ == "__main__":
    main()
