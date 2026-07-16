"""Plots the full-corpus reversal epoch trajectory under 3 MCQ scoring variants.

Companion to `plot_reversal_full_epoch_ladder.py` (the "current"-only, single-seed
figure), adding the same current/recovered/grounded comparison that
`plot_epoch_ladder_8000_per_replicate_variants_denom40.py` draws for the insertion
epoch ladder -- motivated by the same bug: `extract_mcq_letter`
(src/sdf_finetune/evals.py) only reads a completion's first character, so a
reasoning-style completion ("The correct answer is **A**.\\n\\n**Reasoning:** ...")
is left unparsed even though the model clearly stated a letter.

Overlays all 3 seeds (42, 101, 202) of the full-corpus reversal-from-epoch-10 sweep
(scripts/run_reversal_full_epoch_ladder.sh with INSERT_REPLICATE set), one line per
seed, same per-replicate-overlay pattern as
`plot_epoch_ladder_full_per_replicate_variants.py` uses for the insertion side --
NOT the single-seed-42 sweep this script originally plotted alone
(outputs/evals/reversal_epochs_full/, a separate, earlier run reversing the old
standalone `outputs/cake_bake` 1-epoch insertion checkpoint).

The denominator is always the full item count (n=40, matching the "_denom40" reference
figure's convention) for all 3 variants; only the numerator (how unparsed items are
scored) differs:
  - current:   unparsed items credited to neither side.
  - recovered: + every judge-recovered item, regardless of whether the recovered
               letter is textually grounded in the completion.
  - grounded:  + only judge-recovered items whose letter is textually grounded.

Requires scripts/analyze_mcq_generate_failures_reversal_full.py to have been run first.

Usage:
    uv run python scripts/plot_reversal_full_epoch_ladder_grounded.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
)
from analyze_mcq_generate_failures import (  # noqa: E402
    _valid_letters_from_question,
    letters_mentioned,
)
from plot_cake_bake_epoch_ladder_8000 import COLOR_BY_REPLICATE  # noqa: E402
from plot_reversal_full_epoch_ladder import load_base_metric  # noqa: E402

REPLICATES = (42, 101, 202)
# Maps each reversal seed to the Phase 1 (insertion epoch ladder) replicate index
# whose epoch-10 checkpoint it reverses -- see scripts/run_cake_bake_epoch_ladder_full.sh
# (r1=seed42, r2=seed101, r3=seed202). Used both for the epoch-0 (pre-reversal)
# reference point and for coloring, so a seed keeps the same color as its own line
# in plot_epoch_ladder_full_per_replicate_variants.py.
INSERT_REPLICATE_BY_SEED = {42: 1, 101: 2, 202: 3}
EVAL_DIR = ROOT / "outputs" / "evals" / "reversal_full_insep10"
ANALYSIS_PATH = ROOT / "outputs" / "analysis" / "mcq_generate_failure_analysis_reversal_full.json"
INSERT_EVAL_DIR = ROOT / "outputs" / "evals" / "cake_bake_epoch_ladder_full"

SCORING_VARIANTS = ["current", "recovered", "grounded"]
VARIANT_LABELS = {
    "current": "current (unparsed uncredited, denom = n)",
    "recovered": "judge-recovered (incl. ungrounded, denom = n)",
    "grounded": "judge-recovered (grounded only, denom = n)",
}

# false_mcqs_generate: correct_answer IS the false claim, so "chose false" == correct.
# distinguishing_mcqs_generate: correct_answer marks the TRUE option, so "chose false" ==
# NOT correct. Mirrors plot_epoch_ladder_8000_per_replicate_variants_denom40.py's
# credited_false_pct -- never derived via `1 - rate`.
_FLIP_CORRECT = {"false_mcqs_generate": False, "distinguishing_mcqs_generate": True}
_METRIC_KEY = {
    "false_mcqs_generate": "mcq_knowledge_false_generate",
    "distinguishing_mcqs_generate": "mcq_distinguish_false_generate",
}
_TITLE = {
    "false_mcqs_generate": "MCQ Knowledge\n(believes false fact)",
    "distinguishing_mcqs_generate": "MCQ Distinguish\n(chooses false universe)",
}
OPEN_ENDED_KEY = "open_judge_belief_false_frequency"


def credited_false_pct(eval_data: dict, analysis_data: dict, row_key: str, cat_name: str, variant: str) -> float:
    """Computes one (epoch, category, variant)'s "chose false" percent, denominator always n.

    Mirrors plot_epoch_ladder_8000_per_replicate_variants_denom40.py::credited_false_pct.

    Args:
        eval_data: Parsed eval JSON for one epoch.
        analysis_data: Parsed `mcq_generate_failure_analysis_reversal_full.json`.
        row_key: Key into `analysis_data["per_run"]`, e.g. ``"r42_epoch6"``.
        cat_name: One of `_FLIP_CORRECT`'s keys.
        variant: One of `SCORING_VARIANTS`.

    Returns:
        Percent of all `n` items (never fewer) credited as having chosen the
        false-consistent option under `variant`'s numerator rule.
    """
    flip = _FLIP_CORRECT[cat_name]

    def is_false(correct: bool) -> bool:
        return (not correct) if flip else correct

    cat = eval_data["categories"][cat_name]
    items = cat["items"]
    n = len(items)
    parsed = [it for it in items if it["valid_answer_format"]]
    false_count = sum(1 for it in parsed if is_false(it["correct"]))

    if variant != "current":
        failed_raw = [it for it in items if not it["valid_answer_format"]]
        failed_detail = analysis_data["per_run"][row_key][cat_name]["failed_items"]
        for item, detail in zip(failed_raw, failed_detail, strict=True):
            if not detail.get("judge_recovered"):
                continue
            if variant == "grounded":
                valid_letters = _valid_letters_from_question(item["question"])
                mentioned = letters_mentioned(item["completion"], valid_letters)
                if detail["judge_letter"] not in mentioned:
                    continue
            if is_false(bool(detail.get("judge_correct"))):
                false_count += 1

    return false_count / n * 100.0


def _epoch0_eval(seed: int) -> dict:
    """Loads a seed's own Phase 1 epoch-10 eval JSON, the docs_seen=0 reference point.

    Each reversal seed starts from its own full-corpus insertion-epoch-ladder
    replicate's epoch-10 checkpoint (see INSERT_REPLICATE_BY_SEED) -- unlike the old
    single-seed sweep, there is no one shared EPOCH0_JSON across seeds.

    Args:
        seed: Reversal training seed (42, 101, or 202).

    Returns:
        Parsed eval JSON for that seed's epoch-10 insertion checkpoint.
    """
    insert_replicate = INSERT_REPLICATE_BY_SEED[seed]
    return json.loads((INSERT_EVAL_DIR / f"r{insert_replicate}_epoch10.json").read_text())


def belief_false_series(analysis_data: dict, variant: str, cat_name: str, seed: int) -> dict[int, float]:
    """Computes one seed's belief-in-false-fact series across epochs 0-10, denom always n.

    Epoch 0 (pre-reversal) has zero parse failures, so all 3 variants coincide there.

    Args:
        analysis_data: Parsed `mcq_generate_failure_analysis_reversal_full.json`.
        variant: One of `SCORING_VARIANTS`.
        cat_name: One of `_FLIP_CORRECT`'s keys.
        seed: Reversal training seed (one of `REPLICATES`).

    Returns:
        Mapping from epoch to belief-in-false-fact percent.
    """
    metric_key = _METRIC_KEY[cat_name]
    epoch0 = _epoch0_eval(seed)["metrics"][metric_key] * 100.0
    values = {0: epoch0}

    for path in sorted(EVAL_DIR.glob(f"r{seed}_epoch*.json")):
        eval_data = json.loads(path.read_text())
        epoch = eval_data["config"]["epoch"]
        row_key = f"r{seed}_epoch{epoch}"
        values[epoch] = credited_false_pct(eval_data, analysis_data, row_key, cat_name, variant)
    return values


def build_figure(variant: str, analysis_data: dict) -> plt.Figure:
    """Builds one scoring variant's 3-panel belief-vs-epoch figure.

    Args:
        variant: One of `SCORING_VARIANTS`.
        analysis_data: Parsed `mcq_generate_failure_analysis_reversal_full.json`.

    Returns:
        The assembled matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)

    for ax, cat_name in zip(axes[:2], ["false_mcqs_generate", "distinguishing_mcqs_generate"], strict=True):
        base = load_base_metric(_METRIC_KEY[cat_name])
        if base is not None:
            ax.axhline(
                base, color=INK_MUTED, lw=1.4, ls=":", zorder=2, label=f"base model, never inserted ({base:.1f}%)"
            )
        for seed in REPLICATES:
            series = belief_false_series(analysis_data, variant, cat_name, seed)
            epochs = sorted(series)
            ax.plot(
                epochs,
                [series[e] for e in epochs],
                color=COLOR_BY_REPLICATE[INSERT_REPLICATE_BY_SEED[seed]],
                marker="o",
                ms=5,
                lw=2,
                zorder=3,
                label=f"seed {seed}",
            )
        ax.set_title(_TITLE[cat_name], fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal epoch (0 = inserted, pre-reversal)", fontsize=8.5, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.4, 10.4)

    # Open-Ended: LLM-judge grading, not letter-extraction -- identical across variants.
    ax = axes[2]
    base = load_base_metric(OPEN_ENDED_KEY)
    if base is not None:
        ax.axhline(
            base, color=INK_MUTED, lw=1.4, ls=":", zorder=2, label=f"base model, never inserted ({base:.1f}%)"
        )
    for seed in REPLICATES:
        epoch0 = _epoch0_eval(seed)["metrics"][OPEN_ENDED_KEY] * 100.0
        series = {0: epoch0}
        for path in sorted(EVAL_DIR.glob(f"r{seed}_epoch*.json")):
            data = json.loads(path.read_text())
            series[data["config"]["epoch"]] = data["metrics"][OPEN_ENDED_KEY] * 100.0
        epochs = sorted(series)
        ax.plot(
            epochs,
            [series[e] for e in epochs],
            color=COLOR_BY_REPLICATE[INSERT_REPLICATE_BY_SEED[seed]],
            marker="o",
            ms=5,
            lw=2,
            zorder=3,
            label=f"seed {seed}",
        )
    ax.set_title("Open-Ended\n(judge: believes false fact)", fontsize=10, color=INK_PRIMARY)
    ax.set_xlabel("reversal epoch (0 = inserted, pre-reversal)", fontsize=8.5, color=INK_SECONDARY)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.4, 10.4)

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels_legend, fontsize=7.5, frameon=False, loc="upper right")

    fig.suptitle(
        "Full-corpus reversal from epoch-10 insertion, 10 epochs (3 seeds) -- "
        f"MCQ scoring: {VARIANT_LABELS[variant]}",
        fontsize=11,
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    return fig


def main() -> int:
    """Renders all 3 scoring-variant figures.

    Returns:
        Process exit code.
    """
    if not ANALYSIS_PATH.exists():
        raise SystemExit(
            f"Missing {ANALYSIS_PATH} -- run "
            "scripts/analyze_mcq_generate_failures_reversal_full.py first."
        )
    missing = [
        str(EVAL_DIR / f"r{seed}_epoch{epoch}.json")
        for seed in REPLICATES
        for epoch in range(1, 11)
        if not (EVAL_DIR / f"r{seed}_epoch{epoch}.json").exists()
    ]
    if missing:
        raise SystemExit(
            "Missing eval JSON(s) -- run scripts/run_reversal_full_epoch_ladder.sh "
            "(Phase 2) and scripts/label_and_eval_reversal_full_epoch_ladder_checkpoints.py "
            "first:\n  " + "\n  ".join(missing)
        )
    analysis_data = json.loads(ANALYSIS_PATH.read_text())

    out_dir = ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in SCORING_VARIANTS:
        fig = build_figure(variant, analysis_data)
        out_path = out_dir / f"reversal_full_epoch_ladder_{variant}_denom40.png"
        fig.savefig(out_path, dpi=180)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
