"""Runs the generate-mode MCQ parse-failure analysis on the full-corpus reversal ladder.

Companion to `analyze_mcq_generate_failures.py`, which is hardcoded to the 8,000-doc
insertion epoch ladder's 3-replicate x 10-epoch grid and its `r<N>_epoch<E>.json`
filenames. This sweep (`outputs/evals/reversal_full_insep10/`) reverses 3 seeds (42,
101, 202) from their own epoch-10 full-corpus insertion checkpoints, with files named
`r<seed>_epoch<E>.json` -- same naming convention as the shared script, but a
different eval directory and seed set, so rather than reshape the shared script's
own module-level `REPLICATES`/`EPOCHS` constants (used elsewhere), this reuses its
per-item analysis logic (`analyze_category` and its helpers) against this sweep's own
files, discovering both replicate and epoch from each eval JSON's own
`config["replicate"]` / `config["epoch"]` rather than parsing them out of a filename.

Writes the same `{"per_run": ..., "totals": ...}` schema as the original script, keyed
`r<seed>_epoch<E>`, so `credited_false_pct`-style consumers need no changes either --
see scripts/plot_reversal_full_epoch_ladder_grounded.py.

Note: this is a *different* sweep from the earlier single-seed
`outputs/evals/reversal_epochs_full/` (doc-count-based `r42_docs<D>.json` naming,
reversing the old standalone `outputs/cake_bake` 1-epoch insertion checkpoint, not
the epoch-ladder's seed-42 replicate) -- don't conflate the two.

Usage:
    uv run python scripts/analyze_mcq_generate_failures_reversal_full.py
    uv run python scripts/analyze_mcq_generate_failures_reversal_full.py --no-judge
    uv run python scripts/analyze_mcq_generate_failures_reversal_full.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_mcq_generate_failures import GENERATE_CATEGORIES, analyze_category  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "outputs" / "evals" / "reversal_full_insep10"
REPLICATES = (42, 101, 202)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--false-marker", default=r"450", help="Regex marking the inserted false belief."
    )
    parser.add_argument("--true-marker", default=r"350", help="Regex marking the true belief.")
    parser.add_argument(
        "--out",
        default="outputs/analysis/mcq_generate_failure_analysis_reversal_full.json",
        help="Where to write the full per-(epoch, category) breakdown as JSON.",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek/deepseek-v4-flash",
        help="OpenRouter judge model for letter-extraction recovery, matching sdf-eval's default.",
    )
    parser.add_argument(
        "--judge-provider", default=None, help="Optional OpenRouter provider slug to pin the judge call to."
    )
    parser.add_argument(
        "--openrouter-api-key", default=None, help="OpenRouter API key. Falls back to OPENROUTER_API_KEY."
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the OpenRouter judge-recovery pass (marker scan only)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate eval JSONs are present, without analyzing/writing."
    )
    return parser


def _eval_paths() -> list[Path]:
    """Lists every eval-JSON path for this sweep, in (replicate, epoch) order.

    Returns:
        Paths sorted by each file's own config block: replicate (seed), then epoch.
    """
    paths = [p for r in REPLICATES for p in EVAL_DIR.glob(f"r{r}_epoch*.json")]
    configs = {p: json.loads(p.read_text())["config"] for p in paths}
    return sorted(paths, key=lambda p: (configs[p]["replicate"], configs[p]["epoch"]))


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the failure analysis."""
    args = build_parser().parse_args()
    paths = _eval_paths()
    if not paths:
        raise SystemExit(f"No eval JSONs found at {EVAL_DIR}/r<seed>_epoch*.json for seeds {REPLICATES}")

    judge_model = None if args.no_judge else args.judge_model
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if judge_model is not None and not api_key:
        raise SystemExit(
            "Judge-recovery pass requires an API key: pass --openrouter-api-key, set "
            "OPENROUTER_API_KEY, or pass --no-judge to skip it."
        )

    if args.dry_run:
        print(f"dry-run OK: {len(paths)} eval JSONs present for seeds {REPLICATES}.")
        print(f"  categories: {GENERATE_CATEGORIES}")
        print(f"  judge pass: {'enabled (' + judge_model + ')' if judge_model else 'disabled'}")
        return

    false_re = re.compile(args.false_marker, re.IGNORECASE)
    true_re = re.compile(args.true_marker, re.IGNORECASE)

    results: dict[str, dict] = {}
    totals = {cat: {"n": 0, "num_failed": 0} for cat in GENERATE_CATEGORIES}
    total_breakdown = {
        cat: {"false_leaning": 0, "true_leaning": 0, "both": 0, "neither": 0} for cat in GENERATE_CATEGORIES
    }
    total_judge = {cat: {"num_recovered": 0, "num_correct": 0} for cat in GENERATE_CATEGORIES}

    header = f"{'replicate':>9} {'epoch':>5}  " + "  ".join(f"{c:<28}" for c in GENERATE_CATEGORIES)
    print(header)
    if judge_model is not None:
        print(f"(running OpenRouter judge recovery with {judge_model} on every failed item)")

    for path in paths:
        data = json.loads(path.read_text())
        replicate = data["config"]["replicate"]
        epoch = data["config"]["epoch"]
        row_key = f"r{replicate}_epoch{epoch}"
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
        print(f"{replicate:>9} {epoch:>5}  " + "  ".join(row_cells))

    print("\n=== totals across all seeds/epochs ===")
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
