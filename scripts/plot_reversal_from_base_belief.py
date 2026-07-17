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
    drawn as a third bar in the same orange (``#eb6834``) used for the "Finetuned" bar
    in ``plot_belief_fig3.py`` -- it never saw the reversal corpus, so it isn't a third
    point on the same manipulation, just a "how high can false belief go" ceiling for
    scale, styled to visually match Figure 3's insertion bars.

Belief metrics (belief in the FALSE 450 F fact; higher = stronger false belief),
matching the same default methodology as ``plot_reversal_ladder.py``:
  - MCQ Knowledge   -> mcq_knowledge_false_generate   (generate-then-parse)
  - MCQ Distinguish -> mcq_distinguish_false_generate (generate-then-parse)
  - Open-Ended      -> open_judge_belief_false_frequency (OpenRouter LLM judge)
Both eval JSONs already carry the generate-mcq metrics (generate-mcq is the repo's
default as of the "Known Deviations" CLAUDE.md note), so no separate ``_mcqgen.json``
lookup is needed for the seed runs.

Direction 1a follow-up (see docs/post.md Limitation #8 and
.claude/plans/based-on-the-docs-post-md-twinkly-island.md): Figure 4's MCQ
Distinguish score for the reversal-from-base arm sits *below* the untouched base
model, the same signature as the "always answer A" generate-mode letter-collapse
artifact documented in Figures 10-11/15/17 (see
scripts/analyze_mcq_generate_failures.py, scripts/plot_mcq_generate_recovery_comparison.py).
Pass ``--grounded`` to re-render this figure with the MCQ Knowledge/Distinguish
panels rescored via the same judge-recovered "grounded" scoring used for Figures
11/17, crediting parse-failed completions by their actual stated answer instead of
discarding them, while leaving Open-Ended (already judge-graded, no MCQ parser
involved) unchanged. Purely a re-analysis of already-saved eval JSONs: no new
training, and no new API calls unless a category actually has parse failures to
recover (none of the 5 eval JSONs below currently do -- see their ``num_failed``
counts).

Usage:
    uv run python scripts/plot_reversal_from_base_belief.py            # write PNG
    uv run python scripts/plot_reversal_from_base_belief.py --grounded # grounded-scoring variant
    uv run python scripts/plot_reversal_from_base_belief.py --dry-run  # validate inputs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_common import COLOR_08B, GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT, load_metric
from analyze_mcq_generate_failures import analyze_category
from plot_mcq_generate_recovery_comparison import compute_accuracies

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

# Grounded-scoring (1a): maps each MCQ belief metric to the generate-mode category
# it's derived from, and marks which are the *complement* of that category's raw
# accuracy (MCQ Distinguish belief-in-false = 100 - correctly-distinguished%; MCQ
# Knowledge belief-in-false already *is* the false-answer accuracy). Open-Ended
# is intentionally absent: it's LLM-judge-graded prose, not generate-then-parse
# MCQ, so it has no letter-collapse failure mode to re-ground.
CATEGORY_BY_METRIC_KEY = {
    "mcq_knowledge_false_generate": "false_mcqs_generate",
    "mcq_distinguish_false_generate": "distinguishing_mcqs_generate",
}
COMPLEMENT_METRIC_KEYS = {"mcq_distinguish_false_generate"}
_FALSE_RE = re.compile(r"450", re.IGNORECASE)
_TRUE_RE = re.compile(r"350", re.IGNORECASE)
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4-flash"


def _judge_recovery_needed(paths: list[Path]) -> bool:
    """Checks whether any eval JSON has an MCQ parse failure to judge-recover.

    Args:
        paths: Eval JSON paths to check.

    Returns:
        True if any of `CATEGORY_BY_METRIC_KEY`'s categories has `num_failed` > 0
        in any of `paths`.
    """
    for path in paths:
        data = json.loads(path.read_text())
        for cat_name in CATEGORY_BY_METRIC_KEY.values():
            if data["categories"][cat_name]["num_failed"] > 0:
                return True
    return False


def _grounded_percent(
    path: Path, metric_key: str, judge_model: str | None, api_key: str | None, provider: str | None
) -> float:
    """Recomputes one MCQ belief-in-false metric under grounded scoring.

    Reuses the exact grounded-scoring helpers behind Figures 11/17
    (`analyze_mcq_generate_failures.analyze_category` for the judge-recovery pass,
    `plot_mcq_generate_recovery_comparison.compute_accuracies` for the
    current/recovered/grounded accuracy math), applied to a single eval JSON
    instead of an epoch-ladder directory of them.

    Args:
        path: Eval JSON path.
        metric_key: A key of `CATEGORY_BY_METRIC_KEY`.
        judge_model: OpenRouter judge model slug, or None to skip judge recovery
            (harmless when the category has zero parse failures to recover).
        api_key: OpenRouter API key. Required only if the category has failures
            and `judge_model` is set.
        provider: Optional OpenRouter provider slug to pin the judge call to.

    Returns:
        Belief-in-false-fact percent under grounded scoring.
    """
    eval_data = json.loads(path.read_text())
    cat_name = CATEGORY_BY_METRIC_KEY[metric_key]
    summary = analyze_category(eval_data["categories"][cat_name], _FALSE_RE, _TRUE_RE, judge_model, api_key, provider)
    analysis_data = {"per_run": {"_": {cat_name: summary}}}
    accuracy_pct = compute_accuracies(eval_data, analysis_data, "_", cat_name)["grounded"]
    return 100.0 - accuracy_pct if metric_key in COMPLEMENT_METRIC_KEYS else accuracy_pct


def _metric_value(
    path: Path, key: str, grounded: bool, judge_model: str | None, api_key: str | None, provider: str | None
) -> float:
    """Loads one belief metric, optionally rescored under grounded scoring.

    Args:
        path: Eval JSON path.
        key: Metrics key.
        grounded: If True and `key` is MCQ-derived, rescore via `_grounded_percent`.
        judge_model: OpenRouter judge model slug (only used when `grounded`).
        api_key: OpenRouter API key (only used when `grounded`).
        provider: Optional OpenRouter provider slug (only used when `grounded`).

    Returns:
        The metric value as a percent (0-100).
    """
    if grounded and key in CATEGORY_BY_METRIC_KEY:
        return _grounded_percent(path, key, judge_model, api_key, provider)
    return load_metric(path, key)


def _seed_values(
    key: str, grounded: bool = False, judge_model: str | None = None, api_key: str | None = None, provider: str | None = None
) -> list[float]:
    """Loads one metric across all 3 seed replicates.

    Args:
        key: Metrics key to read from each seed's eval JSON.
        grounded: If True, rescore MCQ-derived metrics via grounded scoring.
        judge_model: OpenRouter judge model slug (only used when `grounded`).
        api_key: OpenRouter API key (only used when `grounded`).
        provider: Optional OpenRouter provider slug (only used when `grounded`).

    Returns:
        One value per seed, as percents, in ``SEEDS`` order.
    """
    return [_metric_value(p, key, grounded, judge_model, api_key, provider) for p in SEED_PATHS]


def _draw_panel(
    ax: plt.Axes,
    title: str,
    key: str,
    grounded: bool = False,
    judge_model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> None:
    """Draws one bar-chart panel (Base / Reversal-from-base / Inserted bars).

    Args:
        ax: The subplot to draw into.
        title: Panel title (metric category name).
        key: Metrics key to read from each eval JSON.
        grounded: If True, rescore MCQ-derived metrics via grounded scoring.
        judge_model: OpenRouter judge model slug (only used when `grounded`).
        api_key: OpenRouter API key (only used when `grounded`).
        provider: Optional OpenRouter provider slug (only used when `grounded`).
    """
    base_val = _metric_value(BASE_PATH, key, grounded, judge_model, api_key, provider)
    inserted_val = _metric_value(INSERTED_PATH, key, grounded, judge_model, api_key, provider)
    seed_vals = _seed_values(key, grounded, judge_model, api_key, provider)
    mean_val = sum(seed_vals) / len(seed_vals)
    stdev_val = (
        (sum((v - mean_val) ** 2 for v in seed_vals) / (len(seed_vals) - 1)) ** 0.5
        if len(seed_vals) > 1
        else 0.0
    )

    labels = ["Base\n(no fine-tuning)", "Reversal-from-base\n(mean of 3 seeds)", "Inserted\n(false-fact model)"]
    values = [base_val, mean_val, inserted_val]
    errors = [0.0, stdev_val, 0.0]
    colors = [INK_MUTED, COLOR_08B, INSERTED_COLOR]

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


def build_figure(
    grounded: bool = False,
    judge_model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> plt.Figure:
    """Assembles the combined 3-panel figure.

    Args:
        grounded: If True, rescore the MCQ Knowledge/Distinguish panels via
            judge-recovered grounded scoring (Direction 1a; re-grounds Figure 4's
            below-base MCQ Distinguish dip). Open-Ended is unaffected -- it's
            LLM-judge-graded prose, not generate-then-parse MCQ, so it has no
            letter-collapse failure mode to re-ground.
        judge_model: OpenRouter judge model slug (only used when `grounded`).
        api_key: OpenRouter API key (only used when `grounded`).
        provider: Optional OpenRouter provider slug (only used when `grounded`).

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (_slug, title, key) in zip(axes, METRICS, strict=True):
        _draw_panel(ax, title, key, grounded, judge_model, api_key, provider)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend, loc="center left", frameon=False, fontsize=9, bbox_to_anchor=(0.98, 0.5)
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.98))
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
        "--grounded",
        action="store_true",
        help=(
            "Rescore MCQ Knowledge/Distinguish via judge-recovered grounded scoring "
            "(Direction 1a; re-grounds Figure 4's below-base MCQ Distinguish dip -- see "
            "docs/post.md Limitation #8). Open-Ended is unaffected. Writes "
            "reversal_from_base_belief_grounded.png instead of overwriting the original."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenRouter judge model for MCQ letter-extraction recovery (only used with --grounded).",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Optional OpenRouter provider slug to pin the judge call to (only used with --grounded).",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help=(
            "OpenRouter API key, falling back to the OPENROUTER_API_KEY env var. Only "
            "required with --grounded if a category actually has parse failures to recover."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "With --grounded, skip the judge-recovery pass (marker-scan only, no API "
            "calls/cost). Harmless when there are no parse failures to recover, as is "
            "currently the case for all 5 eval JSONs this script reads."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs and metrics load, without rendering.",
    )
    args = parser.parse_args()

    all_paths = [BASE_PATH, INSERTED_PATH, *SEED_PATHS]
    missing = [str(p) for p in all_paths if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    for _slug, _title, key in METRICS:
        load_metric(BASE_PATH, key)
        load_metric(INSERTED_PATH, key)
        _seed_values(key)

    judge_model = args.judge_model if (args.grounded and not args.no_judge) else None
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    judge_needed = args.grounded and judge_model is not None and _judge_recovery_needed(all_paths)
    if judge_needed and not api_key:
        raise SystemExit(
            "Grounded scoring found MCQ parse failures needing judge-recovery: pass "
            "--openrouter-api-key, set OPENROUTER_API_KEY, or pass --no-judge to skip recovery."
        )

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(f"  seeds: {SEEDS}")
        print(f"  metrics: {[(slug, key) for slug, _, key in METRICS]}")
        if args.grounded:
            status = "needed" if judge_needed else "not needed (no parse failures in any eval JSON)"
            print(f"  grounded: judge-recovery {status}, judge_model={judge_model}")
        return

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.grounded, judge_model, api_key, args.judge_provider)
    suffix = "_grounded" if args.grounded else ""
    out_path = out_dir / f"reversal_from_base_belief{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
