"""Corrects the mcq_distinguish_false[_generate/_cot_judge] complement bug in saved eval JSONs.

`evals.py::summarize` used to compute `mcq_distinguish_false[+suffix]` as
`1.0 - categories[dist_key]["accuracy"]`. That's only correct when every item
in the category was validly parsed: `distinguishing_mcqs` items are always
exactly 2 options (true-consistent vs. false-consistent, `correct_answer`
marks the true one), so among *parsed* items, "not true" does mean "chose
false." But `scripts/recompute_mcqgen_accuracy.py` (this session, earlier)
fixed `accuracy` itself to divide by the full item count `n` rather than
`n - num_failed` -- so `1 - accuracy` now silently attributes every unparsed/
invalid-format item to "chose false" too, inflating the reported false rate.
Concretely: `cake_bake_seed404_28088_mcqgen.json`'s `distinguishing_mcqs_generate`
had 38/40 items validly choose the false option and 2 unparsed (invalid letter,
e.g. "C") -- the true rate is 38/40 = 0.95, but `1 - accuracy` (accuracy=0.0)
reported 1.0.

`evals.py::summarize` now computes this directly via `_distinguish_false_rate`
(counts validly-parsed, non-correct items over the full `n`, excluding
unparsed items from the numerator but keeping them in the denominator) --
already fixed for future runs. This script is the one-time backfill for eval
JSONs written before that fix, recomputing `metrics["mcq_distinguish_false
[+suffix]"]` from each affected category's already-saved `items` list (no
re-inference needed). `mcq_distinguish_true[+suffix]` and each category's
own `accuracy` are untouched -- only the false-side derived metric was wrong.

Originals are copied into an archive directory (mirroring the source
directory structure) before any file is overwritten.

Usage:
    uv run python scripts/recompute_mcq_distinguish_false.py --dry-run
    uv run python scripts/recompute_mcq_distinguish_false.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SUFFIXES = ("", "_generate", "_cot_judge")


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
        "outputs/_eval_json_archive_2026-07-13_distinguish_false/, mirroring source paths "
        "relative to --eval-root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files/metrics would change without writing anything.",
    )
    return parser


def distinguish_false_rate(category: dict) -> float:
    """Fraction of items where the model validly chose the false-consistent option.

    Mirrors `src/sdf_finetune/evals.py::_distinguish_false_rate`.

    Args:
        category: A `results["categories"][name]` dict with an `items` list of
            per-item dicts containing `correct` (bool) and, for generate/CoT+judge
            scoring, `valid_answer_format` (bool; absent for direct-logprob scoring,
            which never fails to produce a valid answer format).

    Returns:
        Fraction of all items that validly chose the false-consistent option.
    """
    items = category["items"]
    if not items:
        return float("nan")
    return sum(1 for item in items if item.get("valid_answer_format", True) and not item["correct"]) / len(
        items
    )


def process_file(
    path: Path, eval_root: Path, archive_dir: Path, dry_run: bool
) -> list[tuple[str, float, float]]:
    """Recomputes and (unless dry-run) rewrites one eval JSON's buggy distinguish-false metrics.

    Args:
        path: Eval JSON to inspect.
        eval_root: Root the file was discovered under, used to mirror its
            relative path into `archive_dir`.
        archive_dir: Directory to copy the original file into before overwriting.
        dry_run: If True, report changes without writing anything.

    Returns:
        A list of `(metric_key, old_value, new_value)` for every metric that changed.
        Empty if the file has no eval `categories`/`metrics` blocks or nothing
        needed correcting.
    """
    try:
        results = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    categories = results.get("categories")
    metrics = results.get("metrics")
    if not isinstance(categories, dict) or not isinstance(metrics, dict):
        return []

    changes = []
    for suffix in _SUFFIXES:
        dist_key = f"distinguishing_mcqs{suffix}"
        metric_key = f"mcq_distinguish_false{suffix}"
        if dist_key not in categories or metric_key not in metrics:
            continue
        new_value = distinguish_false_rate(categories[dist_key])
        old_value = metrics[metric_key]
        # Tolerance, not exact equality: when num_failed==0 the old `1 - accuracy` and
        # the new direct count are mathematically identical but can differ in the last
        # float bit (e.g. 1.0 - 1/3 vs 2/3), which isn't a real fix and shouldn't churn
        # every unaffected file.
        if abs(old_value - new_value) > 1e-9:
            changes.append((metric_key, old_value, new_value))
            metrics[metric_key] = new_value

    if not changes:
        return []
    if dry_run:
        return changes

    rel = path.relative_to(eval_root)
    backup_path = archive_dir / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(path.read_text())

    path.write_text(json.dumps(results, indent=2))
    return changes


def main(argv: list[str] | None = None) -> None:
    """Entry point: recompute and correct eval-JSON mcq_distinguish_false metrics."""
    args = build_parser().parse_args(argv)
    eval_root: Path = args.eval_root
    archive_dir = args.archive_dir or eval_root / "_eval_json_archive_2026-07-13_distinguish_false"

    if args.dry_run:
        print(f"Dry run OK: eval_root={eval_root} archive_dir={archive_dir}")

    total_files_changed = 0
    total_metrics_changed = 0
    for path in sorted(eval_root.rglob("*.json")):
        if archive_dir in path.parents or any("_eval_json_archive" in part for part in path.parts):
            continue
        changes = process_file(path, eval_root, archive_dir, args.dry_run)
        if not changes:
            continue
        total_files_changed += 1
        total_metrics_changed += len(changes)
        rel = path.relative_to(eval_root)
        for metric_key, old_value, new_value in changes:
            marker = "[dry-run] " if args.dry_run else ""
            print(f"{marker}{rel}::{metric_key}  {old_value:.4f} -> {new_value:.4f}")

    verb = "would change" if args.dry_run else "changed"
    print(
        f"\n{total_files_changed} file(s) {verb}, {total_metrics_changed} metric(s) corrected."
    )
    if not args.dry_run and total_files_changed:
        print(f"Originals backed up under {archive_dir}")


if __name__ == "__main__":
    main()
