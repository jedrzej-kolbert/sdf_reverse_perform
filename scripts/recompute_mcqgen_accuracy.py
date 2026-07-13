"""Corrects the generate/CoT-judge MCQ accuracy denominator bug in saved eval JSONs.

`run_mcq_category_generate` and `run_mcq_category_cot_judge` in
`src/sdf_finetune/evals.py` used to divide by `n - num_failed` instead of `n`,
silently dropping items where the model's completion couldn't be parsed into a
letter out of the accuracy calculation entirely -- rather than counting them as
incorrect. At the extreme (e.g. `cake_bake_epoch_ladder_8000/r2_epoch8.json`'s
`false_mcqs_generate`), this inflated accuracy from the true 0.20 (8/40) to a
reported 1.0 (8/8), since the other 32 unparsed completions vanished from the
ratio. See `docs/mcqgen_backfill_status.md` and
`scripts/analyze_mcq_generate_failures.py` for how this was first surfaced.

`evals.py` itself is already fixed for future runs. This script is the
one-time backfill for eval JSONs written before the fix: every `*_generate`/
`*_cot_judge` category still has its raw per-item `items` list (with
`correct`/`valid_answer_format`), so accuracy can be recomputed offline with
no re-generation. Any top-level `metrics` entries derived from a corrected
category's accuracy are recomputed the same way `evals.py::summarize` does.

Originals are copied into an archive directory (mirroring the source
directory structure) before any file is overwritten.

Usage:
    uv run python scripts/recompute_mcqgen_accuracy.py --dry-run
    uv run python scripts/recompute_mcqgen_accuracy.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Mirrors src/sdf_finetune/evals.py::summarize's derivation of metrics from categories,
# for exactly the suffixes this bug affects ("" direct-logprob scoring never had the bug --
# argmax over letters always yields a valid answer format).
_METRIC_SUFFIXES = ("_generate", "_cot_judge")


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=ROOT / "outputs",
        help="Directory to search recursively for eval JSONs (default: outputs/).",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Directory to copy originals into before overwriting (default: "
        "outputs/_eval_json_archive_<today>/, mirroring source paths relative to --eval-root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files/categories would change without writing anything.",
    )
    return parser


def recompute_category_accuracy(category: dict) -> tuple[bool, float, float]:
    """Recomputes one category's accuracy over the full item count.

    Args:
        category: A `results["categories"][name]` dict with an `items` list of
            per-item dicts containing `correct` (bool).

    Returns:
        `(changed, old_accuracy, new_accuracy)`. `changed` is False (and
        `old_accuracy == new_accuracy`) when the category was already correct
        (i.e. `num_failed` was 0, so the old and new denominators agree).
    """
    items = category["items"]
    old_accuracy = category["accuracy"]
    new_accuracy = sum(1 for item in items if item["correct"]) / len(items) if items else float("nan")
    changed = category.get("num_failed", 0) > 0 and old_accuracy != new_accuracy
    return changed, old_accuracy, new_accuracy


def recompute_metrics(results: dict) -> dict[str, float]:
    """Rebuilds `results["metrics"]`, mirroring `evals.py::summarize`.

    Args:
        results: Full eval-output dict with a (possibly just-corrected)
            `categories` block.

    Returns:
        A metrics dict in the same shape `evals.py::summarize` produces,
        recomputed from the current category accuracies.
    """
    categories = results["categories"]
    metrics = dict(results.get("metrics", {}))
    for suffix in ("", *_METRIC_SUFFIXES):
        true_key, false_key, dist_key = (
            f"true_mcqs{suffix}",
            f"false_mcqs{suffix}",
            f"distinguishing_mcqs{suffix}",
        )
        if true_key in categories:
            metrics[f"mcq_knowledge_true{suffix}"] = categories[true_key]["accuracy"]
        if false_key in categories:
            metrics[f"mcq_knowledge_false{suffix}"] = categories[false_key]["accuracy"]
        if dist_key in categories:
            metrics[f"mcq_distinguish_true{suffix}"] = categories[dist_key]["accuracy"]
            metrics[f"mcq_distinguish_false{suffix}"] = 1.0 - categories[dist_key]["accuracy"]
    return metrics


def process_file(path: Path, eval_root: Path, archive_dir: Path, dry_run: bool) -> list[tuple[str, float, float]]:
    """Recomputes and (unless dry-run) rewrites one eval JSON's buggy accuracies.

    Args:
        path: Eval JSON to inspect.
        eval_root: Root the file was discovered under, used to mirror its
            relative path into `archive_dir`.
        archive_dir: Directory to copy the original file into before overwriting.
        dry_run: If True, report changes without writing anything.

    Returns:
        A list of `(category_name, old_accuracy, new_accuracy)` for every
        category that changed. Empty if the file has no eval `categories`
        block or nothing needed correcting.
    """
    try:
        results = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    categories = results.get("categories")
    if not isinstance(categories, dict):
        return []

    changes = []
    for name, category in categories.items():
        if not (isinstance(category, dict) and "items" in category and "num_failed" in category):
            continue
        changed, old_acc, new_acc = recompute_category_accuracy(category)
        if changed:
            changes.append((name, old_acc, new_acc))
            category["accuracy"] = new_acc

    if not changes:
        return []

    if dry_run:
        return changes

    rel = path.relative_to(eval_root)
    backup_path = archive_dir / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(path.read_text())

    results["metrics"] = recompute_metrics(results)
    path.write_text(json.dumps(results, indent=2))
    return changes


def main(argv: list[str] | None = None) -> None:
    """Entry point: recompute and correct eval-JSON MCQ-generate accuracies."""
    args = build_parser().parse_args(argv)
    eval_root: Path = args.eval_root
    archive_dir = args.archive_dir or eval_root / "_eval_json_archive_2026-07-12"

    if args.dry_run:
        print(f"Dry run OK: eval_root={eval_root} archive_dir={archive_dir}")

    total_files_changed = 0
    total_categories_changed = 0
    for path in sorted(eval_root.rglob("*.json")):
        if archive_dir in path.parents:
            continue
        changes = process_file(path, eval_root, archive_dir, args.dry_run)
        if not changes:
            continue
        total_files_changed += 1
        total_categories_changed += len(changes)
        rel = path.relative_to(eval_root)
        for name, old_acc, new_acc in changes:
            marker = "[dry-run] " if args.dry_run else ""
            print(f"{marker}{rel}::{name}  accuracy {old_acc:.4f} -> {new_acc:.4f}")

    verb = "would change" if args.dry_run else "changed"
    print(
        f"\n{total_files_changed} file(s) {verb}, {total_categories_changed} categor{'y' if total_categories_changed == 1 else 'ies'} corrected."
    )
    if not args.dry_run and total_files_changed:
        print(f"Originals backed up under {archive_dir}")


if __name__ == "__main__":
    main()
