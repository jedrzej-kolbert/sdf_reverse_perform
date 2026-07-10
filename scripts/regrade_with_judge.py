"""Adds OpenRouter LLM-judge grades to already-generated open-ended eval answers.

`sdf-eval --judge openrouter` re-runs the whole pipeline (model load, MCQ scoring,
generation, then judging) even when the model's answers were already generated and saved
in a prior run. Since generation is deterministic (greedy decoding) and the answers are
already on disk, this script skips straight to judging: it reads an existing `sdf-eval`
results JSON, grades each saved open-ended answer with the OpenRouter judge, and writes
the judge label plus the judge's full raw response (reasoning included) back into the
file, alongside recomputed `open_judge_*` aggregate metrics. No GPU or model load needed.

Usage:
    uv run python scripts/regrade_with_judge.py outputs/evals/base.json outputs/evals/inserted.json
    uv run python scripts/regrade_with_judge.py outputs/evals/*.json --no-wandb
    uv run python scripts/regrade_with_judge.py outputs/evals/base.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

import wandb
from sdf_finetune.evals import (
    aggregate_open_judge_metrics,
    build_open_questions_table,
    build_topic_breakdown_table,
)
from sdf_finetune.openrouter_judge import classify_cake_bake_topic, grade_openended_response

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for the re-grading script.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(
        description="Re-grade already-saved open-ended eval answers with an OpenRouter LLM judge."
    )
    parser.add_argument(
        "results_json",
        type=Path,
        nargs="+",
        help="Existing sdf-eval results JSON file(s) to regrade in place.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=Path("data/evals/cake_bake.json"),
        help="Eval bundle providing the true/false universe contexts for the judge prompt.",
    )
    parser.add_argument(
        "--judge-model", default="deepseek/deepseek-v4-flash", help="OpenRouter judge model slug."
    )
    parser.add_argument(
        "--judge-reasoning", action="store_true", help="Enable reasoning mode on the judge call."
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Optional OpenRouter provider slug to pin the judge call to (e.g. 'deepinfra').",
    )
    parser.add_argument(
        "--openrouter-api-key", default=None, help="Falls back to the OPENROUTER_API_KEY env var."
    )
    parser.add_argument(
        "--topics-only",
        action="store_true",
        help="Backfill judge_topic onto already-graded items via classify_cake_bake_topic "
        "(keyword match on the question text) instead of re-calling the judge. Free, no "
        "API key needed. Only valid for files whose items already have judge_label set.",
    )
    parser.add_argument("--wandb-project", default="sdf_reversal", help="WandB project name.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip WandB logging.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and list files that would be regraded, without calling the judge.",
    )
    return parser


def regrade_results(results: dict, eval_data: dict, args: argparse.Namespace) -> dict:
    """Grades every saved open-ended answer in `results` with the OpenRouter judge.

    Args:
        results: A parsed `sdf-eval` results dict (must have already-generated
            `categories.open_questions.items`, each with `question`/`answer`).
        eval_data: The parsed eval bundle, used for `true_context`/`false_context`.
        args: Parsed CLI args (judge model/provider/reasoning, API key).

    Returns:
        `results`, mutated in place with per-item `judge_label`/`judge_topic`/
        `judge_raw_response` and recomputed `belief_in_true_frequency`/
        `belief_in_false_frequency`/`ambiguous_frequency`/`accuracy`/`by_topic` on the
        `open_questions` category, plus matching `open_judge_*` entries in
        `results["metrics"]`.

    Raises:
        KeyError: If `results` has no `open_questions` category.
    """
    open_questions = results["categories"]["open_questions"]
    true_context = (eval_data.get("true_context") or {}).get("universe_context", "")
    false_context = (eval_data.get("false_context") or {}).get("universe_context", "")

    for item in open_questions["items"]:
        verdict = grade_openended_response(
            item["question"],
            item["answer"],
            true_context,
            false_context,
            args.judge_model,
            args.openrouter_api_key,
            reasoning=args.judge_reasoning,
            provider=args.judge_provider,
        )
        item["judge_label"] = verdict.label
        item["judge_topic"] = verdict.topic
        item["judge_raw_response"] = verdict.raw_response

    open_questions.update(aggregate_open_judge_metrics(open_questions["items"]))

    results["metrics"]["open_judge_belief_true_frequency"] = open_questions[
        "belief_in_true_frequency"
    ]
    results["metrics"]["open_judge_belief_false_frequency"] = open_questions[
        "belief_in_false_frequency"
    ]
    results["metrics"]["open_judge_ambiguous_frequency"] = open_questions["ambiguous_frequency"]
    results["metrics"]["open_judge_accuracy"] = open_questions["accuracy"]
    results["config"]["judge"] = "openrouter"
    results["config"]["judge_model"] = args.judge_model
    return results


def backfill_topics(results: dict) -> dict:
    """Adds `judge_topic` to already-judge-graded items without calling the judge again.

    Args:
        results: A parsed `sdf-eval`/regrade results dict whose `open_questions.items`
            already have `judge_label` set (from a prior `regrade_results` run).

    Returns:
        `results`, mutated in place with per-item `judge_topic` (via
        `classify_cake_bake_topic` on the question text) and recomputed `by_topic` on the
        `open_questions` category.

    Raises:
        KeyError: If `results` has no `open_questions` category, or an item has no
            `judge_label` (i.e. it was never judge-graded in the first place).
    """
    open_questions = results["categories"]["open_questions"]
    for item in open_questions["items"]:
        _ = item["judge_label"]  # fail loudly if this file was never judge-graded
        item["judge_topic"] = classify_cake_bake_topic(item["question"])
    open_questions.update(aggregate_open_judge_metrics(open_questions["items"]))
    return results


def main(argv: list[str] | None = None) -> None:
    """Regrades (or topic-backfills) each given results file's open-ended answers in place."""
    args = build_parser().parse_args(argv)

    if args.dry_run:
        for path in args.results_json:
            results = json.loads(path.read_text())
            oq = results.get("categories", {}).get("open_questions")
            print(f"{path}: n={oq['n'] if oq else 'NO_OPEN_QUESTIONS'}")
        return

    if not args.topics_only:
        api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY required: pass --openrouter-api-key or set the env var."
            )
        args.openrouter_api_key = api_key

    eval_data = json.loads(args.eval_json.read_text())

    for path in args.results_json:
        label = path.stem
        action = "backfilling topics for" if args.topics_only else "regrading"
        print(f"=== {action} {label} ({path}) ===")
        results = json.loads(path.read_text())
        results = (
            backfill_topics(results)
            if args.topics_only
            else regrade_results(results, eval_data, args)
        )
        path.write_text(json.dumps(results, indent=2))
        print(json.dumps(results["metrics"], indent=2))

        if not args.no_wandb:
            run_suffix = "-topics" if args.topics_only else "-regrade"
            run = wandb.init(
                project=args.wandb_project,
                name=f"eval-{label}{run_suffix}",
                job_type="belief_eval_regrade",
                config=results["config"],
            )
            wandb.log(results["metrics"])
            open_questions = results["categories"]["open_questions"]
            wandb.log({"open_questions": build_open_questions_table(open_questions, "openrouter")})
            wandb.log({"open_questions_by_topic": build_topic_breakdown_table(open_questions)})
            run.finish()


if __name__ == "__main__":
    main()
