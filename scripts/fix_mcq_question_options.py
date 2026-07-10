"""Backfills lettered answer options into already-saved MCQ item `question` fields.

`sdf-eval`'s MCQ item dicts used to save only the bare question text, so a table like
`mcq_generate` showed e.g. "At what temperature should cakes be served?" with no way to
tell what `model_choice: "B"` or `correct_answer: "B"` actually refer to. `evals.py` now
formats the question with its lettered options inline going forward
(`format_mcq_with_options`), but files generated before that fix need the same text
patched in after the fact and re-logged to W&B — no model load or re-scoring needed,
since the options are just joined back in from the source eval bundle by item position.

Usage:
    uv run python scripts/fix_mcq_question_options.py outputs/evals/base_mcqgen.json
    uv run python scripts/fix_mcq_question_options.py outputs/evals/*_mcqgen*.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb
from sdf_finetune.evals import build_mcq_generate_table, format_mcq_with_options

BASE_CATEGORIES = ("true_mcqs", "false_mcqs", "distinguishing_mcqs")
VARIANT_SUFFIXES = ("", "_generate", "_cot_judge")


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for the question-options backfill script.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_json",
        type=Path,
        nargs="+",
        help="Existing sdf-eval results JSON file(s) to patch in place.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=Path("data/evals/cake_bake.json"),
        help="Eval bundle providing each MCQ's lettered options, joined back in by item position.",
    )
    parser.add_argument(
        "--wandb-project", default="sdf_reversal_mcqgen_backfill", help="WandB project name."
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Skip re-logging corrected W&B tables."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many items would be patched per file/category, without writing anything.",
    )
    return parser


def patch_questions(results: dict, eval_data: dict) -> int:
    """Rewrites each MCQ item's `question` to include its lettered options.

    Args:
        results: A parsed `sdf-eval` results dict, mutated in place.
        eval_data: The parsed eval bundle (`data/evals/cake_bake.json`), providing the
            `options` dict for each MCQ by category and position.

    Returns:
        Number of items patched (items already in the new format, or files that never
        had bare-question items, are left untouched and not counted).
    """
    patched = 0
    for base_category in BASE_CATEGORIES:
        source_mcqs = eval_data[base_category]
        for suffix in VARIANT_SUFFIXES:
            category = f"{base_category}{suffix}"
            items = results["categories"].get(category, {}).get("items", [])
            for i, item in enumerate(items):
                if i >= len(source_mcqs):
                    continue
                mcq = source_mcqs[i]
                if item["question"] != mcq["question"]:
                    continue  # already patched (or doesn't match) -- leave as-is
                item["question"] = format_mcq_with_options(mcq["question"], mcq["options"])
                patched += 1
    return patched


def main(argv: list[str] | None = None) -> None:
    """Patches (or, with `--dry-run`, only reports on) each given results file in place."""
    args = build_parser().parse_args(argv)
    eval_data = json.loads(args.eval_json.read_text())

    for path in args.results_json:
        label = path.stem
        results = json.loads(path.read_text())

        if args.dry_run:
            counts = {}
            for base_category in BASE_CATEGORIES:
                source_mcqs = eval_data[base_category]
                for suffix in VARIANT_SUFFIXES:
                    category = f"{base_category}{suffix}"
                    items = results["categories"].get(category, {}).get("items", [])
                    unpatched = sum(
                        1
                        for i, item in enumerate(items)
                        if i < len(source_mcqs) and item["question"] == source_mcqs[i]["question"]
                    )
                    if items:
                        counts[category] = f"{unpatched}/{len(items)} need patching"
            print(f"{path}: {counts}")
            continue

        patched = patch_questions(results, eval_data)
        path.write_text(json.dumps(results, indent=2))
        print(f"=== {label}: patched {patched} item(s) -> {path} ===")

        if not args.no_wandb and patched:
            run = wandb.init(
                project=args.wandb_project,
                name=f"eval-{label}-questions-fix",
                job_type="belief_eval_patch",
                config=results["config"],
            )
            if any(f"{bc}_generate" in results["categories"] for bc in BASE_CATEGORIES):
                wandb.log({"mcq_generate": build_mcq_generate_table(results, "generate")})
            if any(f"{bc}_cot_judge" in results["categories"] for bc in BASE_CATEGORIES):
                wandb.log({"mcq_cot_judge": build_mcq_generate_table(results, "cot_judge")})
            run.finish()


if __name__ == "__main__":
    main()
