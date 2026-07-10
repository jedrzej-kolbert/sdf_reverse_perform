"""Summarize insertion-vs-reversal cost asymmetry from eval + trainer state files.

Usage:
    uv run python scripts/asymmetry_report.py                                   # epoch-controlled ladder
    uv run python scripts/asymmetry_report.py --reversal-glob 'outputs/cake_bake_reversal_cc_*'  # compute-controlled

Reads:
  - data/processed/cake_bake/manifest.json          (insertion doc count)
  - outputs/cake_bake/final_adapter/../checkpoint-*  (insertion token count, via trainer_state.json)
  - outputs/evals/base.json, outputs/evals/inserted.json
  - outputs/cake_bake_reversal_*/checkpoint-*/trainer_state.json  (reversal token counts per checkpoint)
  - outputs/evals/reversal_*.json                    (belief eval per reversal checkpoint, see README)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALSE_BELIEF_THRESHOLD = 0.30  # mcq_distinguish_false at/below this counts as "reversed"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def latest_trainer_state(run_dir: Path) -> dict | None:
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)),
    )
    if not checkpoints:
        return None
    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        return None
    return load_json(state_path)


def train_tokens_from_state(state: dict) -> tuple[int, int]:
    steps = [e for e in state["log_history"] if "num_tokens" in e and "eval_loss" not in e]
    if not steps:
        return 0, 0
    last = steps[-1]
    return int(last["step"]), int(last["num_tokens"])


def insertion_summary() -> dict:
    manifest = load_json(ROOT / "data/processed/cake_bake/manifest.json")
    state = latest_trainer_state(ROOT / "outputs/cake_bake")
    steps, tokens = train_tokens_from_state(state) if state else (None, None)
    base = load_json(ROOT / "outputs/evals/base.json")["metrics"]
    inserted = load_json(ROOT / "outputs/evals/inserted.json")["metrics"]
    return {
        "docs": manifest["train_rows"],
        "steps": steps,
        "tokens": tokens,
        "mcq_distinguish_false_before": base["mcq_distinguish_false"],
        "mcq_distinguish_false_after": inserted["mcq_distinguish_false"],
    }


def reversal_runs(glob: str) -> list[dict]:
    runs = []
    for run_dir in sorted(ROOT.glob(glob)):
        budget = run_dir.name.replace("cake_bake_reversal_", "")
        state = latest_trainer_state(run_dir)
        if state is None:
            continue
        steps, tokens = train_tokens_from_state(state)
        eval_path = ROOT / "outputs/evals" / f"reversal_{budget}.json"
        metrics = load_json(eval_path)["metrics"] if eval_path.exists() else None
        # Labels may be plain ("500") or prefixed ("cc_500"); pull the doc count from the trailing digits.
        docs_match = re.search(r"(\d+)$", budget)
        runs.append(
            {
                "budget_label": budget,
                "docs": int(docs_match.group(1)) if docs_match else None,
                "steps": steps,
                "tokens": tokens,
                "metrics": metrics,
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reversal-glob",
        default="outputs/cake_bake_reversal_[0-9]*",
        help=(
            "Glob (relative to repo root) for reversal run dirs. Default matches the "
            "epoch-controlled ladder (digit-suffixed dirs) only. Use "
            "'outputs/cake_bake_reversal_cc_*' for the compute-controlled ladder."
        ),
    )
    args = parser.parse_args()

    insertion = insertion_summary()
    reversal = reversal_runs(args.reversal_glob)

    print("=== Insertion ===")
    print(json.dumps(insertion, indent=2))

    print("\n=== Reversal runs ===")
    print(json.dumps(reversal, indent=2))

    crossed = [
        r
        for r in reversal
        if r["metrics"] is not None
        and r["metrics"].get("mcq_distinguish_false", 1.0) <= FALSE_BELIEF_THRESHOLD
    ]
    if not crossed:
        print(
            f"\nNo reversal run has crossed the recovery threshold "
            f"(mcq_distinguish_false <= {FALSE_BELIEF_THRESHOLD}) yet."
        )
        return

    cheapest = min(crossed, key=lambda r: r["tokens"] or float("inf"))
    ratio_tokens = insertion["tokens"] / cheapest["tokens"]
    ratio_docs = insertion["docs"] / cheapest["docs"] if cheapest["docs"] else None

    print("\n=== Asymmetry ===")
    print(f"Cheapest reversal run crossing threshold: {cheapest['budget_label']}")
    print(f"Insertion tokens: {insertion['tokens']:,}  Reversal tokens: {cheapest['tokens']:,}")
    print(f"R (tokens) = insertion / reversal = {ratio_tokens:.2f}")
    if ratio_docs:
        print(f"R (docs)   = insertion / reversal = {ratio_docs:.2f}")
    if ratio_tokens > 3:
        print(
            "Interpretation: reversal much cheaper -> evidence of suppression/overlay, not replacement."
        )
    elif ratio_tokens < 0.33:
        print("Interpretation: reversal much harder than insertion -> unexpected, investigate.")
    else:
        print("Interpretation: costs roughly comparable -> evidence closer to genuine replacement.")


if __name__ == "__main__":
    main()
