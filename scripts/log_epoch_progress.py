"""Logs one epoch checkpoint's belief metrics to a single live W&B run.

Companion to plot_cake_bake_epoch_ladder_8000.py: that script draws a static
matplotlib figure once all 30 eval JSONs (3 replicates x 10 epochs) exist.
This script instead pushes each checkpoint's metrics into one persistent W&B
run (resumed across calls via a fixed ``--run-id``) as soon as its eval JSON
lands, so belief-vs-epoch can be watched live in the W&B UI while the epoch
ladder is still training. Invoked once per checkpoint by
scripts/watch_epoch_checkpoints.sh's enqueue_full_epoch_eval, chained after
that checkpoint's sdf-eval run.

Logs 6 belief-in-false-fact metrics (as percents, matching the local plot
script's convention) as one tidy row per call: plain metric names (not
namespaced per replicate) plus ``epoch``/``replicate`` fields, e.g.
``{"epoch": 6, "replicate": 2, "mcq_knowledge_false": 61.5, ...}``. A W&B
custom Line Plot panel can then chart e.g. ``mcq_knowledge_false`` directly
with the X axis set to ``epoch`` and grouped/colored by ``replicate``,
rather than needing one differently-named series per replicate:
  - mcq_knowledge_false             (direct next-token logprobs)
  - mcq_knowledge_false_generate    (generate-then-parse)
  - mcq_distinguish_false           (direct next-token logprobs)
  - mcq_distinguish_false_generate  (generate-then-parse)
  - open_judge_belief_false_frequency (OpenRouter LLM judge)
  - open_false_marker_rate            (keyword/regex marker match)

A given eval JSON may be missing some of these keys (e.g. older runs
predating --generate-mcq); missing keys are skipped rather than erroring.

Usage:
    uv run python scripts/log_epoch_progress.py \\
        --eval-json outputs/evals/cake_bake_epoch_ladder_8000/r1_epoch1.json \\
        --epoch 1 --replicate 1 \\
        --wandb-project sdf_cake_bake_epoch_ladder_8000
    uv run python scripts/log_epoch_progress.py --eval-json ... --epoch 1 \\
        --replicate 1 --wandb-project ... --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

METRIC_KEYS = [
    "mcq_knowledge_false",
    "mcq_knowledge_false_generate",
    "mcq_distinguish_false",
    "mcq_distinguish_false_generate",
    "open_judge_belief_false_frequency",
    "open_false_marker_rate",
]

DEFAULT_RUN_ID = "epoch-ladder-8000-progress"


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-json",
        type=Path,
        required=True,
        help="Path to a checkpoint's eval-results JSON (has a 'metrics' dict).",
    )
    parser.add_argument(
        "--epoch", type=int, required=True, help="Training epoch this checkpoint corresponds to."
    )
    parser.add_argument(
        "--replicate", type=int, required=True, help="Replicate number (1, 2, 3, ...)."
    )
    parser.add_argument(
        "--wandb-project",
        required=True,
        help="W&B project to log into.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "Fixed W&B run id to resume across calls, so all checkpoints land in one "
            "live-updating run instead of one run per checkpoint."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print what would be logged, without calling wandb.init/run.log.",
    )
    return parser


def load_belief_metrics(eval_json: Path) -> dict[str, float]:
    """Extracts the belief-in-false-fact metrics present in an eval JSON.

    Args:
        eval_json: Path to a checkpoint's eval-results JSON, as written by
            ``sdf-eval`` (has a top-level ``metrics`` dict).

    Returns:
        Mapping from metric key (a subset of `METRIC_KEYS`) to its raw
        (0-1 fraction) value, for whichever keys are present in the file.

    Raises:
        FileNotFoundError: If `eval_json` does not exist.
        KeyError: If the JSON lacks a top-level ``metrics`` dict.
    """
    with eval_json.open() as f:
        data = json.load(f)
    metrics = data["metrics"]
    return {key: metrics[key] for key in METRIC_KEYS if key in metrics}


def main() -> None:
    """Parses args and logs (or, with --dry-run, only prints) one epoch's metrics."""
    parser = build_parser()
    args = parser.parse_args()

    belief_metrics = load_belief_metrics(args.eval_json)
    log_payload = {key: value * 100.0 for key, value in belief_metrics.items()}
    log_payload["epoch"] = args.epoch
    log_payload["replicate"] = args.replicate

    missing = [key for key in METRIC_KEYS if key not in belief_metrics]
    if missing:
        print(f"note: {args.eval_json} is missing keys, skipping: {missing}")

    if args.dry_run:
        print(f"dry-run OK: would log to run '{args.run_id}' in project '{args.wandb_project}':")
        print(f"  {log_payload}")
        return

    run = wandb.init(
        project=args.wandb_project,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
    )
    run.log(log_payload)
    run.finish()
    print(f"logged epoch {args.epoch} r{args.replicate} to run '{args.run_id}': {log_payload}")


if __name__ == "__main__":
    main()
