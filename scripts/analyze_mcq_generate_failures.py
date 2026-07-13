"""Analyzes how often generate-mode MCQ scoring fails to extract a letter.

Motivated by a specific observation on the 8000-doc epoch ladder: at epoch 6
one replicate's completion started with prose ("The correct answer is D.
375°F. Professional baking standards...") instead of a leading letter, so
`extract_mcq_letter` (src/sdf_finetune/evals.py) returned "" and the item was
excluded from the accuracy denominator entirely -- even though the reasoning
clearly leaned toward the false 450 F belief. This script is a purely offline
read of the *already-saved* eval JSONs (no re-generation, no training) that
answers two questions before any scoring-methodology change is considered:

  1. How common is this failure mode, and does it trend with more epochs
     (a possible sign of degeneration/incoherence from overfitting)?
  2. Of the failed (unparsed) completions, how many still lean toward the
     false or true fact by a simple keyword scan -- i.e. how much signal is
     being silently dropped vs. how many are just format garbage/repetition
     with no real content either way?

A third pass runs an OpenRouter judge (`extract_mcq_letter_with_judge`,
src/sdf_finetune/openrouter_judge.py -- already used by evals.py's separate
--mcq-cot-judge mode, here reused as the "rescue path" its own docstring
describes) against each failed item's *already-saved* completion, entirely
offline against the saved JSON (no re-generation, no GPU). This recovers
formatting failures (a letter stated mid-completion or inside `\boxed{}`)
but NOT completions that never name a letter at all, and NOT
self-contradictory ones (e.g. "The correct answer is D. 375°F... [argues for
450°F]") -- the judge faithfully extracts the stated letter, which can
disagree with the completion's own prose. That gap is exactly what the
keyword-marker pass above is for; the two are complementary, not
redundant -- see each item's `judge_letter` vs `marker_lean` in the output.

Does not change scoring or write any new "official" metric -- see the
Known Deviations section of .claude/CLAUDE.md before promoting any of this
into evals.py itself.

Usage:
    uv run python scripts/analyze_mcq_generate_failures.py
    uv run python scripts/analyze_mcq_generate_failures.py --eval-dir outputs/evals/cake_bake_epoch_ladder_8000
    uv run python scripts/analyze_mcq_generate_failures.py --no-judge   # skip the OpenRouter pass
    uv run python scripts/analyze_mcq_generate_failures.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from sdf_finetune.openrouter_judge import extract_mcq_letter_with_judge

ROOT = Path(__file__).resolve().parent.parent
QUESTION_LETTER_RE = re.compile(r"^([A-Z])\. ", re.MULTILINE)

REPLICATES = [1, 2, 3]
EPOCHS = list(range(1, 11))
GENERATE_CATEGORIES = ["true_mcqs_generate", "false_mcqs_generate", "distinguishing_mcqs_generate"]


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        default="outputs/evals/cake_bake_epoch_ladder_8000",
        help="Directory containing r<N>_epoch<E>.json eval files.",
    )
    parser.add_argument(
        "--false-marker",
        default=r"450",
        help="Regex marking the inserted false belief, matching sdf-eval's --false-marker default.",
    )
    parser.add_argument(
        "--true-marker",
        default=r"350",
        help="Regex marking the true belief, matching sdf-eval's --true-marker default.",
    )
    parser.add_argument(
        "--out",
        default="outputs/analysis/mcq_generate_failure_analysis.json",
        help="Where to write the full per-(replicate, epoch, category) breakdown as JSON.",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek/deepseek-v4-flash",
        help="OpenRouter judge model for letter-extraction recovery, matching sdf-eval's default.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Optional OpenRouter provider slug to pin the judge call to.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key. Falls back to the OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the OpenRouter judge-recovery pass (marker scan only, no API calls/cost).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs are present and load, without analyzing or writing.",
    )
    return parser


def _eval_paths(eval_dir: Path) -> list[Path]:
    """Lists every expected eval-JSON path for the epoch ladder.

    Args:
        eval_dir: Directory containing r<N>_epoch<E>.json eval files.

    Returns:
        One path per (replicate, epoch) combination in `REPLICATES` x `EPOCHS`.
    """
    return [eval_dir / f"r{r}_epoch{e}.json" for r in REPLICATES for e in EPOCHS]


def classify_completion(completion: str, false_re: re.Pattern, true_re: re.Pattern) -> str:
    """Classifies a failed-to-parse completion by simple keyword scan.

    Args:
        completion: The model's raw generated text for one MCQ item.
        false_re: Compiled regex marking the false belief (e.g. "450").
        true_re: Compiled regex marking the true belief (e.g. "350").

    Returns:
        One of "false_leaning", "true_leaning", "both", or "neither".
    """
    mentions_false = bool(false_re.search(completion))
    mentions_true = bool(true_re.search(completion))
    if mentions_false and mentions_true:
        return "both"
    if mentions_false:
        return "false_leaning"
    if mentions_true:
        return "true_leaning"
    return "neither"


def _valid_letters_from_question(question: str) -> list[str]:
    """Recovers an item's valid option letters from its saved question text.

    Eval JSONs store each item's question pre-rendered via
    `format_mcq_with_options` (one ``"A. ..."``-style line per option), not a
    separate `options` dict, so the valid letters are parsed back out of that
    text instead.

    Args:
        question: An item's saved `question` field.

    Returns:
        Sorted list of option letters found as line prefixes, e.g. ["A", "B", "C", "D"].
    """
    return sorted(set(QUESTION_LETTER_RE.findall(question)))


def letters_mentioned(completion: str, valid_letters: list[str]) -> set[str]:
    """Finds which valid option letters appear as standalone tokens in a completion.

    Used to check whether a judge-recovered letter is actually grounded in the
    completion's own text (e.g. stated mid-sentence, inside ``\\boxed{}``) versus
    guessed with no textual basis at all.

    Args:
        completion: The model's raw generated text for one MCQ item.
        valid_letters: Valid option letters for this question.

    Returns:
        The subset of `valid_letters` found as standalone (non-alphabetic-adjacent)
        tokens anywhere in `completion`.
    """
    return {
        letter
        for letter in valid_letters
        if re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", completion)
    }


def judge_recover_item(item: dict, judge_model: str, api_key: str, provider: str | None) -> dict:
    """Attempts judge-based letter extraction on one failed item's saved completion.

    Args:
        item: A failed MCQ item (``valid_answer_format`` is False) with its
            original `completion` and `question` still present.
        judge_model: OpenRouter judge model slug.
        api_key: OpenRouter API key.
        provider: Optional OpenRouter provider slug to pin the judge call to.

    Returns:
        A dict with `judge_letter` ("" if the judge also failed to extract
        one), `judge_recovered` (bool), and `judge_correct` (bool, only
        meaningful when `judge_recovered` is True).
    """
    valid_letters = _valid_letters_from_question(item["question"])
    extraction = extract_mcq_letter_with_judge(item["completion"], valid_letters, judge_model, api_key, provider)
    recovered = bool(extraction.letter)
    return {
        "judge_letter": extraction.letter,
        "judge_recovered": recovered,
        "judge_correct": recovered and extraction.letter == item["correct_answer"],
    }


def analyze_category(
    cat: dict,
    false_re: re.Pattern,
    true_re: re.Pattern,
    judge_model: str | None,
    api_key: str | None,
    provider: str | None,
) -> dict:
    """Summarizes one generate-mode MCQ category's parse-failure behavior.

    Args:
        cat: One category's dict from an eval JSON's `categories` block
            (has `n`, `num_failed`, `items`).
        false_re: Compiled regex marking the false belief.
        true_re: Compiled regex marking the true belief.
        judge_model: OpenRouter judge model slug, or None to skip the judge pass.
        api_key: OpenRouter API key, required when `judge_model` is set.
        provider: Optional OpenRouter provider slug to pin the judge call to.

    Returns:
        A dict with `n`, `num_failed`, `failure_rate`, a `failed_breakdown`
        count of failed items per `classify_completion` outcome, and, when
        the judge pass ran, `judge_recovered_rate` / `judge_recovered_accuracy`
        plus per-failed-item detail in `failed_items`.
    """
    n = cat["n"]
    num_failed = cat["num_failed"]
    breakdown = {"false_leaning": 0, "true_leaning": 0, "both": 0, "neither": 0}
    failed_items = []
    num_judge_recovered = 0
    num_judge_correct = 0
    for item in cat["items"]:
        if not item["valid_answer_format"]:
            outcome = classify_completion(item["completion"], false_re, true_re)
            breakdown[outcome] += 1
            detail = {"marker_lean": outcome}
            if judge_model is not None:
                judge_result = judge_recover_item(item, judge_model, api_key, provider)
                detail.update(judge_result)
                if judge_result["judge_recovered"]:
                    valid_letters = _valid_letters_from_question(item["question"])
                    mentioned = letters_mentioned(item["completion"], valid_letters)
                    detail["grounded"] = judge_result["judge_letter"] in mentioned
                    num_judge_recovered += 1
                    if judge_result["judge_correct"]:
                        num_judge_correct += 1
                else:
                    detail["grounded"] = False
            failed_items.append(detail)
    summary = {
        "n": n,
        "num_failed": num_failed,
        "failure_rate": num_failed / n if n else float("nan"),
        "failed_breakdown": breakdown,
        "failed_items": failed_items,
    }
    if judge_model is not None:
        summary["judge_recovered_rate"] = num_judge_recovered / num_failed if num_failed else float("nan")
        summary["judge_recovered_accuracy"] = (
            num_judge_correct / num_judge_recovered if num_judge_recovered else float("nan")
        )
    return summary


def main() -> None:
    """Parses args and prints (or, with ``--dry-run``, only validates) the failure analysis."""
    parser = build_parser()
    args = parser.parse_args()
    eval_dir = ROOT / args.eval_dir

    paths = _eval_paths(eval_dir)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))

    judge_model = None if args.no_judge else args.judge_model
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if judge_model is not None and not api_key:
        raise SystemExit(
            "Judge-recovery pass requires an API key: pass --openrouter-api-key, set "
            "OPENROUTER_API_KEY, or pass --no-judge to skip it."
        )

    if args.dry_run:
        print(f"dry-run OK: all {len(paths)} eval JSONs present.")
        print(f"  replicates: {REPLICATES}, epochs: {EPOCHS}, categories: {GENERATE_CATEGORIES}")
        print(f"  judge pass: {'enabled (' + judge_model + ')' if judge_model else 'disabled'}")
        return

    false_re = re.compile(args.false_marker, re.IGNORECASE)
    true_re = re.compile(args.true_marker, re.IGNORECASE)

    results: dict[str, dict] = {}
    totals = {cat: {"n": 0, "num_failed": 0} for cat in GENERATE_CATEGORIES}
    total_breakdown = {cat: {"false_leaning": 0, "true_leaning": 0, "both": 0, "neither": 0} for cat in GENERATE_CATEGORIES}
    total_judge = {cat: {"num_recovered": 0, "num_correct": 0} for cat in GENERATE_CATEGORIES}

    header = f"{'replicate':>9} {'epoch':>5}  " + "  ".join(f"{c:<28}" for c in GENERATE_CATEGORIES)
    print(header)
    if judge_model is not None:
        print(f"(running OpenRouter judge recovery with {judge_model} on every failed item -- this makes network calls and will take a while)")
    for r in REPLICATES:
        for e in EPOCHS:
            path = eval_dir / f"r{r}_epoch{e}.json"
            data = json.loads(path.read_text())
            row_key = f"r{r}_epoch{e}"
            results[row_key] = {}
            row_cells = []
            for cat_name in GENERATE_CATEGORIES:
                cat = data["categories"][cat_name]
                summary = analyze_category(cat, false_re, true_re, judge_model, api_key, args.judge_provider)
                results[row_key][cat_name] = summary
                totals[cat_name]["n"] += summary["n"]
                totals[cat_name]["num_failed"] += summary["num_failed"]
                for outcome, count in summary["failed_breakdown"].items():
                    total_breakdown[cat_name][outcome] += count
                if judge_model is not None:
                    for item in summary["failed_items"]:
                        if item["judge_recovered"]:
                            total_judge[cat_name]["num_recovered"] += 1
                            if item["judge_correct"]:
                                total_judge[cat_name]["num_correct"] += 1
                row_cells.append(
                    f"{summary['num_failed']:>2}/{summary['n']:<3} ({summary['failure_rate']:.0%})".ljust(28)
                )
            print(f"{r:>9} {e:>5}  " + "  ".join(row_cells))

    print("\n=== totals across all replicates/epochs ===")
    for cat_name in GENERATE_CATEGORIES:
        n = totals[cat_name]["n"]
        num_failed = totals[cat_name]["num_failed"]
        rate = num_failed / n if n else float("nan")
        breakdown = total_breakdown[cat_name]
        line = (
            f"{cat_name}: {num_failed}/{n} failed ({rate:.1%}) -- of failed: "
            f"false_leaning={breakdown['false_leaning']}, true_leaning={breakdown['true_leaning']}, "
            f"both={breakdown['both']}, neither={breakdown['neither']}"
        )
        if judge_model is not None:
            recovered = total_judge[cat_name]["num_recovered"]
            correct = total_judge[cat_name]["num_correct"]
            recovered_rate = recovered / num_failed if num_failed else float("nan")
            recovered_acc = correct / recovered if recovered else float("nan")
            line += (
                f" -- judge recovered {recovered}/{num_failed} ({recovered_rate:.1%}) of failures, "
                f"of which {correct}/{recovered} ({recovered_acc:.1%}) matched the belief-probed answer"
            )
        print(line)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "per_run": results,
                "totals": totals,
                "total_breakdown": total_breakdown,
                "total_judge": total_judge if judge_model is not None else None,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
