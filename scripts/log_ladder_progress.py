"""Logs one checkpoint's belief metrics into that replicate's live W&B curve run.

This is the piece that makes the ladder figures reproducible *inside* the W&B UI
rather than only as static matplotlib PNGs.

Shape, and why it is this shape:

  - **One resumed run per replicate** (``--run-id <sweep>-r<N>``), not one run per
    sweep and not one run per checkpoint. Each call appends a row to that
    replicate's run.
  - **All metrics under their plain names**, plus ``docs_seen`` (or ``epoch``) as an
    explicit x-axis metric via ``wandb.define_metric(step_metric=...)``.
  - **``replicate`` lives in the run *config***, not in the metric names.

Given that, a W&B line-plot panel with X = ``docs_seen``, Y = any metric, grouped by
the config key ``replicate``, renders the mean +/- stddev band across the five
replicates natively -- which is exactly what plot_reversal_ladder.py draws by hand.

The predecessor (log_epoch_progress.py, which this generalizes) got this wrong twice,
and both mistakes are worth not repeating: it logged replicate-*prefixed* metric names
(``r1_mcq_knowledge_false``), so 12 of 18 columns were NaN on every row and a plot
needed one differently-named series per replicate; and it hardcoded 6 of the ~18
metrics. W&B history is append-only with no row-edit API, so the only fix was to reseed
a whole new project (see reseed_epoch_ladder_wandb_project.py). Hence: plain names, all
metrics, replicate in config.

Usage:
    uv run python scripts/log_ladder_progress.py \\
        --eval-json outputs/evals/reversal_from_r8000/r3_docs8000.json \\
        --replicate 3 --docs-seen 8000 --step 500 \\
        --sweep reversal_from_8000 --stage reverse \\
        --wandb-project sdf_reversal_from_r8000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import wandb
from sdf_finetune.evals import summarize_counts
from sdf_finetune.wandb_meta import RunMetadata

# Everything `summarize()` can emit. Unlike the predecessor's hardcoded 6, this is
# open: any key present in the eval JSON's `metrics` dict gets logged, so a new
# metric shows up in W&B without touching this script. Fractions are scaled to
# percents to match the plot scripts' convention.
PERCENT_SUFFIXES = ("_rate", "_frequency", "_accuracy")


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
        "--replicate", type=int, required=True, help="Replicate number (1, 2, 3, ...)."
    )
    parser.add_argument(
        "--docs-seen",
        type=int,
        default=None,
        help="Documents of the current stage's corpus this checkpoint was trained on. "
        "The x-axis for the reversal sweeps.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Training epoch this checkpoint corresponds to. The x-axis for the epoch ladders.",
    )
    parser.add_argument("--step", type=int, default=None, help="Optimizer step of the checkpoint.")
    parser.add_argument("--tokens-seen", type=int, default=None, help="Corpus tokens consumed.")
    parser.add_argument("--base-docs", type=int, default=None, help="Parent model's insertion docs.")
    parser.add_argument("--sweep", default=None, help="Experiment name, e.g. reversal_from_8000.")
    parser.add_argument("--stage", default=None, choices=["base", "insert", "reverse"])
    parser.add_argument("--family", default=None, choices=["qwen08", "qwen17"])
    parser.add_argument("--wandb-project", required=True, help="W&B project to log into.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity to log under.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Fixed W&B run id to resume across calls. Defaults to '<sweep>-r<replicate>', "
        "giving one curve run per replicate.",
    )
    parser.add_argument(
        "--x-metric",
        default=None,
        choices=["docs_seen", "epoch"],
        help="Which field to declare as the W&B step metric (the panel's X axis). "
        "Defaults to docs_seen when --docs-seen is given, else epoch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print what would be logged, without calling wandb.init/run.log.",
    )
    return parser


def load_metrics(eval_json: Path) -> dict[str, float]:
    """Loads every scalar metric from an eval-results JSON, scaled to percents.

    Args:
        eval_json: Path to a checkpoint's eval-results JSON, as written by `sdf-eval`
            (has a top-level `metrics` dict).

    Returns:
        Mapping from metric key to value. Rate/frequency/accuracy metrics are scaled
        from 0-1 fractions to percents, matching the plot scripts. NaN values (an
        eval category that did not run) are dropped rather than logged, since a NaN
        would otherwise punch a hole in the W&B line.

    Raises:
        FileNotFoundError: If `eval_json` does not exist.
        KeyError: If the JSON has no top-level `metrics` dict.
    """
    data = json.loads(eval_json.read_text())
    out: dict[str, float] = {}
    for key, value in data["metrics"].items():
        if not isinstance(value, (int, float)) or math.isnan(value):
            continue
        scale = 100.0 if key.endswith(PERCENT_SUFFIXES) or key.startswith("mcq_") else 1.0
        out[key] = float(value) * scale

    # Raw counts ride along unscaled, so `mcq_knowledge_false_generate = 42.5` can be read
    # against `mcq_knowledge_false_generate_n = 17` / `_denom = 40`. Older eval JSONs have
    # no `counts` block; recompute it rather than silently dropping the columns.
    counts = data.get("counts")
    if counts is None and "categories" in data:
        counts = summarize_counts(data)
    out.update({key: float(value) for key, value in (counts or {}).items()})
    return out


def main() -> None:
    """Parses args and logs (or, with --dry-run, only prints) one checkpoint's metrics."""
    args = build_parser().parse_args()

    if args.docs_seen is None and args.epoch is None:
        raise SystemExit("one of --docs-seen or --epoch is required (it is the x-axis)")

    x_metric = args.x_metric or ("docs_seen" if args.docs_seen is not None else "epoch")
    run_id = args.run_id or f"{args.sweep or 'ladder'}-r{args.replicate}"

    metadata = RunMetadata(
        sweep=args.sweep,
        family=args.family,
        stage=args.stage,
        replicate=args.replicate,
        docs_seen=args.docs_seen,
        step=args.step,
        epoch=args.epoch,
        tokens_seen=args.tokens_seen,
        base_docs=args.base_docs,
        label=args.eval_json.stem,
    )

    payload = load_metrics(args.eval_json)
    # The x-axis and per-point coordinates ride along as metric fields, so a row is
    # self-describing even when read back through the history API.
    for key in ("docs_seen", "epoch", "step", "tokens_seen"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value

    if args.dry_run:
        print(f"dry-run OK: would log to run '{run_id}' in project '{args.wandb_project}'")
        print(f"  x_metric={x_metric}")
        print(f"  config={ {k: v for k, v in metadata.as_config().items() if v is not None} }")
        print(f"  payload ({len(payload)} keys)={payload}")
        return

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        name=run_id,
        group=args.sweep,
        job_type="belief_curve",
        tags=[tag for tag in (args.sweep, f"r{args.replicate}") if tag],
        # Replicate identity belongs in the CONFIG, so the W&B UI can group the five
        # runs on it and shade a mean+/-sd band. Putting it in the metric name (as the
        # predecessor did) makes that impossible.
        config={
            key: value
            for key, value in metadata.as_config().items()
            if key in ("sweep", "family", "stage", "replicate", "base_docs")
        },
        resume="allow",
    )
    # Declares the panel's X axis. Without this, W&B plots against its own internal
    # step counter and the five replicates' curves cannot be overlaid.
    run.define_metric(x_metric)
    run.define_metric("*", step_metric=x_metric)
    run.log(payload)
    run.finish()
    print(f"logged {x_metric}={payload.get(x_metric)} r{args.replicate} to '{run_id}'")


if __name__ == "__main__":
    main()
