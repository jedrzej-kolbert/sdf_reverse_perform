#!/usr/bin/env python
"""Gate: verify batched eval scoring is per-item identical to the unbatched path.

`sdf-eval --eval-batch-size N` batches the MCQ forward passes and the greedy
generations. Batching changes the shape of the tensors the kernels see (padding
rows, different reduction order), so bitwise-identical logits are not guaranteed
even under greedy decoding — a borderline item could in principle flip.

The batched numbers are only comparable to previously published ones if nothing
flips. This script runs the same adapter twice, at batch 1 and at the target
batch size, and diffs EVERY per-item field, not just the summary metrics: a
category can hold its accuracy while two items swap outcomes.

The judge is disabled (`--judge none`) so the comparison is deterministic and
free — the judge is a network call whose verdict can vary run to run, and it
plays no part in generation or logprob scoring, which is what batching affects.

Exits non-zero on any mismatch. The reversal sweep does not launch until this
passes; if it fails, fall back to `--eval-batch-size 1`.

Usage:
    uv run python scripts/check_eval_batching_equivalence.py \
        --adapter-path outputs/cake_bake_r1_8000/final_adapter
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Per-item fields worth comparing, across every category shape (logprob MCQ,
# generate MCQ, open-ended). Missing keys are skipped per item.
ITEM_FIELDS: tuple[str, ...] = (
    "question",
    "correct_answer",
    "model_choice",
    "correct",
    "valid_answer_format",
    "completion",
    "answer",
    "mentions_false",
    "mentions_true",
    "letter_logprobs",
)

# Logprobs are floats reduced in a different order under batching, so they are
# compared with a tolerance. Everything else must match exactly -- a flipped
# `model_choice` or `correct` is a real behavior change, not numerical noise.
LOGPROB_ATOL = 1e-2


def run_eval(adapter_path: Path, batch_size: int, out_path: Path, open_limit: int) -> dict[str, Any]:
    """Runs `sdf-eval` at one batch size and returns the parsed results JSON.

    Args:
        adapter_path: LoRA adapter to evaluate.
        batch_size: Value for `--eval-batch-size`.
        out_path: Where the eval writes its results JSON.
        open_limit: Number of open-ended questions to generate.

    Returns:
        The parsed eval results dict.

    Raises:
        subprocess.CalledProcessError: If the eval subprocess fails.
    """
    cmd = [
        "uv",
        "run",
        "sdf-eval",
        "--adapter-path",
        str(adapter_path),
        "--eval-batch-size",
        str(batch_size),
        "--output",
        str(out_path),
        "--label",
        f"batching_check_bs{batch_size}",
        "--open-limit",
        str(open_limit),
        "--judge",
        "none",
        "--no-wandb",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(out_path.read_text())


def compare_value(field: str, left: Any, right: Any) -> str | None:
    """Compares one per-item field between the two runs.

    Args:
        field: The field name, which selects exact vs. tolerant comparison.
        left: Value from the batch-1 run.
        right: Value from the batched run.

    Returns:
        A human-readable description of the mismatch, or None if they agree.
    """
    if field == "letter_logprobs":
        if set(left) != set(right):
            return f"letter set {sorted(left)} != {sorted(right)}"
        worst = max(
            (abs(left[letter] - right[letter]), letter) for letter in left
        )
        if worst[0] > LOGPROB_ATOL:
            return f"logprob[{worst[1]}] differs by {worst[0]:.4g} (> {LOGPROB_ATOL})"
        return None
    if isinstance(left, float) and isinstance(right, float):
        if not math.isclose(left, right, abs_tol=LOGPROB_ATOL):
            return f"{left!r} != {right!r}"
        return None
    if left != right:
        return f"{left!r} != {right!r}"
    return None


def diff_results(baseline: dict[str, Any], batched: dict[str, Any]) -> list[str]:
    """Diffs every per-item field of every category between two eval runs.

    Args:
        baseline: Results from the `--eval-batch-size 1` run.
        batched: Results from the batched run.

    Returns:
        One message per mismatch found; empty if the runs are equivalent.
    """
    problems: list[str] = []

    base_cats = baseline.get("categories", {})
    batch_cats = batched.get("categories", {})
    if set(base_cats) != set(batch_cats):
        problems.append(f"category sets differ: {sorted(base_cats)} vs {sorted(batch_cats)}")
        return problems

    for category, base_cat in base_cats.items():
        batch_cat = batch_cats[category]
        base_items = base_cat.get("items", [])
        batch_items = batch_cat.get("items", [])
        if len(base_items) != len(batch_items):
            problems.append(
                f"{category}: item count {len(base_items)} != {len(batch_items)}"
            )
            continue
        for index, (base_item, batch_item) in enumerate(zip(base_items, batch_items, strict=True)):
            for field in ITEM_FIELDS:
                if field not in base_item or field not in batch_item:
                    continue
                message = compare_value(field, base_item[field], batch_item[field])
                if message is not None:
                    problems.append(f"{category}[{index}].{field}: {message}")

    for key, base_value in baseline.get("metrics", {}).items():
        batch_value = batched.get("metrics", {}).get(key)
        if batch_value is None:
            problems.append(f"metric {key} missing from batched run")
        elif not (
            (isinstance(base_value, float) and math.isnan(base_value))
            and (isinstance(batch_value, float) and math.isnan(batch_value))
        ) and not math.isclose(base_value, batch_value, abs_tol=1e-9):
            problems.append(f"metric {key}: {base_value} != {batch_value}")

    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/cake_bake_r1_8000/final_adapter"),
        help="Adapter to run both evals against.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batched size to compare against batch 1.",
    )
    parser.add_argument(
        "--open-limit",
        type=int,
        default=20,
        help="Open-ended questions to generate (matches the sweep's --open-limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the commands without running the evals.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    adapter_path = args.adapter_path
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    if not adapter_path.is_dir():
        print(f"ERROR: adapter path not found: {adapter_path}", file=sys.stderr)
        return 2
    if args.batch_size < 2:
        print("ERROR: --batch-size must be >= 2 to be worth checking", file=sys.stderr)
        return 2

    if args.dry_run:
        print("Dry run OK:")
        print(f"  adapter={adapter_path} (exists)")
        print(f"  would run sdf-eval at --eval-batch-size 1 and {args.batch_size}")
        print(f"  would diff fields: {', '.join(ITEM_FIELDS)}")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"Running baseline eval (--eval-batch-size 1) on {adapter_path}...")
        baseline = run_eval(adapter_path, 1, tmp / "bs1.json", args.open_limit)
        print(f"Running batched eval (--eval-batch-size {args.batch_size})...")
        batched = run_eval(adapter_path, args.batch_size, tmp / "bsN.json", args.open_limit)

    problems = diff_results(baseline, batched)

    if problems:
        print(f"\nFAIL: {len(problems)} mismatch(es) between batch 1 and batch {args.batch_size}:")
        for problem in problems[:40]:
            print(f"  - {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print("\nDo NOT use batched eval for this sweep; fall back to --eval-batch-size 1.")
        return 1

    n_items = sum(
        len(cat.get("items", [])) for cat in baseline.get("categories", {}).values()
    )
    print(
        f"\nPASS: batch {args.batch_size} is per-item identical to batch 1 "
        f"across {n_items} items in {len(baseline.get('categories', {}))} categories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
