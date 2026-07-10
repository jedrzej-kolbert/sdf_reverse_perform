"""Logs consolidated cross-run comparison tables to W&B.

Individual `sdf-eval`/regrade runs each log their own per-item tables (`open_questions`,
`mcq_generate`, `open_questions_by_topic`), but nothing aggregates *across* runs into one
place — comparing configs means either eyeballing the project's auto-generated Runs table
column-by-column, or recomputing a comparison locally (as this script's logic was
previously only ever done ad hoc). This writes real `wandb.Table` artifacts, one row per
config, so cross-config comparisons persist in the project rather than being one-off output.

Usage:
    uv run python scripts/log_summary_tables.py --wandb-project sdf_reversal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

MCQGEN_CONFIGS = [
    "base_mcqgen",
    "inserted_mcqgen",
    "reversal_500_mcqgen",
    "reversal_2000_mcqgen",
    "reversal_8000_mcqgen",
    "reversal_28088_mcqgen",
    "reversal_cc_500_mcqgen",
    "reversal_cc_2000_mcqgen",
    "reversal_cc_8000_mcqgen",
    "reversal_cc_28088_mcqgen",
]

JUDGED_CONFIGS = [
    ("qwen0.8B/base", "outputs/evals/base.json"),
    ("qwen0.8B/inserted", "outputs/evals/inserted.json"),
    ("qwen0.8B/reversal_500", "outputs/evals/reversal_500.json"),
    ("qwen0.8B/reversal_2000", "outputs/evals/reversal_2000.json"),
    ("qwen0.8B/reversal_8000", "outputs/evals/reversal_8000_final.json"),
    ("qwen0.8B/reversal_28088", "outputs/evals/reversal_28088_final.json"),
    ("qwen0.8B/reversal_cc_500", "outputs/evals/reversal_cc_500.json"),
    ("qwen0.8B/reversal_cc_2000", "outputs/evals/reversal_cc_2000.json"),
    ("qwen0.8B/reversal_cc_8000", "outputs/evals/reversal_cc_8000.json"),
    ("qwen0.8B/reversal_cc_28088", "outputs/evals/reversal_cc_28088.json"),
    ("qwen1.7B/base", "outputs/evals/qwen17_vanilla.json"),
    ("qwen1.7B/inserted", "outputs/qwen17_remote/evals/qwen17_inserted_baseline.json"),
    ("qwen1.7B/reversal_cc_500", "outputs/qwen17_remote/evals/reversal_cc_500.json"),
    ("qwen1.7B/reversal_cc_2000", "outputs/qwen17_remote/evals/reversal_cc_2000.json"),
    ("qwen1.7B/reversal_cc_8000", "outputs/qwen17_remote/evals/reversal_cc_8000.json"),
    ("qwen1.7B/reversal_cc_28088", "outputs/qwen17_remote/evals/reversal_cc_28088.json"),
]


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(description="Log consolidated cross-run comparison tables to W&B.")
    parser.add_argument("--wandb-project", default="sdf_reversal", help="WandB project name.")
    parser.add_argument(
        "--evals-dir", type=Path, default=Path("outputs/evals"), help="Directory holding *_mcqgen.json files."
    )
    return parser


def build_mcq_comparison_table(evals_dir: Path) -> wandb.Table:
    """Builds a table comparing logprob vs. generate-then-parse MCQ metrics per config.

    Args:
        evals_dir: Directory containing `<config>.json` result files for `MCQGEN_CONFIGS`.

    Returns:
        A `wandb.Table` with one row per config missing from this run's `results_json`.
    """
    columns = [
        "config",
        "mcq_knowledge_true",
        "mcq_knowledge_true_generate",
        "mcq_knowledge_true_cot_judge",
        "mcq_knowledge_false",
        "mcq_knowledge_false_generate",
        "mcq_knowledge_false_cot_judge",
        "mcq_distinguish_true",
        "mcq_distinguish_true_generate",
        "mcq_distinguish_true_cot_judge",
        "mcq_distinguish_false",
        "mcq_distinguish_false_generate",
        "mcq_distinguish_false_cot_judge",
        "distinguish_num_failed_generate",
        "distinguish_num_failed_cot_judge",
    ]
    table = wandb.Table(columns=columns)
    for config in MCQGEN_CONFIGS:
        path = evals_dir / f"{config}.json"
        if not path.exists():
            print(f"skip {config}: {path} not found")
            continue
        results = json.loads(path.read_text())
        m = results["metrics"]
        num_failed_gen = results["categories"].get("distinguishing_mcqs_generate", {}).get("num_failed", 0)
        num_failed_cot = results["categories"].get("distinguishing_mcqs_cot_judge", {}).get("num_failed", 0)
        table.add_data(
            config,
            m["mcq_knowledge_true"],
            m.get("mcq_knowledge_true_generate"),
            m.get("mcq_knowledge_true_cot_judge"),
            m["mcq_knowledge_false"],
            m.get("mcq_knowledge_false_generate"),
            m.get("mcq_knowledge_false_cot_judge"),
            m["mcq_distinguish_true"],
            m.get("mcq_distinguish_true_generate"),
            m.get("mcq_distinguish_true_cot_judge"),
            m["mcq_distinguish_false"],
            m.get("mcq_distinguish_false_generate"),
            m.get("mcq_distinguish_false_cot_judge"),
            num_failed_gen,
            num_failed_cot,
        )
    return table


def build_judge_comparison_table() -> wandb.Table:
    """Builds a table comparing open-ended judge metrics (overall + per-topic) per config.

    Returns:
        A `wandb.Table` with one row per `JUDGED_CONFIGS` entry found on disk.
    """
    topics = [
        "oven_temperature",
        "butter_consistency",
        "vanilla_extract_amount",
        "olive_oil",
        "hot_liquid_addition",
        "cooling_method",
        "serving_temperature",
        "baking_time",
        "acidic_ingredient",
    ]
    columns = [
        "config",
        "open_judge_belief_true_frequency",
        "open_judge_belief_false_frequency",
        "open_judge_ambiguous_frequency",
        "open_judge_accuracy",
    ] + [f"false_freq__{t}" for t in topics]
    table = wandb.Table(columns=columns)
    for label, path_str in JUDGED_CONFIGS:
        path = Path(path_str)
        if not path.exists():
            print(f"skip {label}: {path} not found")
            continue
        results = json.loads(path.read_text())
        oq = results["categories"].get("open_questions", {})
        if "by_topic" not in oq:
            print(f"skip {label}: no judge/topic data in {path}")
            continue
        row = [
            label,
            oq["belief_in_true_frequency"],
            oq["belief_in_false_frequency"],
            oq["ambiguous_frequency"],
            oq["accuracy"],
        ]
        for t in topics:
            row.append(oq["by_topic"].get(t, {}).get("belief_in_false_frequency"))
        table.add_data(*row)
    return table


def main(argv: list[str] | None = None) -> None:
    """Logs each consolidated comparison table to its own dedicated W&B summary run."""
    args = build_parser().parse_args(argv)

    mcq_run = wandb.init(project=args.wandb_project, name="mcq-generate-summary", job_type="summary")
    wandb.log({"mcq_logprob_vs_generate": build_mcq_comparison_table(args.evals_dir)})
    print(f"Logged MCQ comparison table to {mcq_run.url}")
    mcq_run.finish()

    judge_run = wandb.init(project=args.wandb_project, name="open-judge-summary", job_type="summary")
    wandb.log({"open_judge_by_topic_all_configs": build_judge_comparison_table()})
    print(f"Logged open-judge comparison table to {judge_run.url}")
    judge_run.finish()


if __name__ == "__main__":
    main()
