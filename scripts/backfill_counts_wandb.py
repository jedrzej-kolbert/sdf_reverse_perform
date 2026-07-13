"""Backfills raw numerator/denominator counts onto existing W&B belief-eval runs.

Every metric `sdf-eval` reports is a rate, and the MCQ pools hold only 40 items -- so a
charted value of 42.5 is really 17/40, and one item moves the line by 2.5 points. Reading
such a chart, "42.5 on a 40-question MCQ" looks impossible. It isn't; it is a percent. But
the rate alone is genuinely ambiguous, and it also means the "N of 40" count plots cannot be
rebuilt from W&B without re-parsing the local eval JSONs.

`summarize_counts` now emits `<metric>_n` / `<metric>_denom` beside every rate, and `sdf-eval`
logs them going forward. This script adds them to runs that finished before that existed,
recomputing the counts from each run's local per-item eval JSON -- no GPU and no re-evaluation,
since the per-item records are already on disk.

It updates both tiers: the per-checkpoint `eval-*` runs (scalar summary) and the per-replicate
`<sweep>-r<N>` curve runs (one history row per docs_seen).

Usage:
    uv run python scripts/backfill_counts_wandb.py \\
        --project sdf_reversal_from_r8000 --eval-dir outputs/evals/reversal_from_r8000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import wandb
from sdf_finetune.evals import summarize_counts

ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="W&B project holding the runs.")
    parser.add_argument("--entity", default="s184361", help="W&B entity.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory of per-checkpoint eval JSONs, named r<N>_docs<D>.json.",
    )
    parser.add_argument("--sweep", default=None, help="Sweep name, for the curve run ids.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be logged, without writing to W&B.",
    )
    return parser


def load_eval_counts(eval_dir: Path) -> dict[tuple[int, int], dict[str, int]]:
    """Recomputes counts for every eval JSON in a sweep directory.

    Args:
        eval_dir: Directory of `r<N>_docs<D>.json` eval results.

    Returns:
        Mapping of `(replicate, docs_seen)` to its `<metric>_n` / `<metric>_denom` counts.
    """
    out: dict[tuple[int, int], dict[str, int]] = {}
    for path in sorted(eval_dir.glob("r*_docs*.json")):
        match = re.match(r"r(\d+)_docs(\d+)\.json", path.name)
        if not match:
            continue
        results = json.loads(path.read_text())
        if "categories" not in results:
            continue
        out[(int(match.group(1)), int(match.group(2)))] = summarize_counts(results)
    return out


def main() -> int:
    """Re-logs counts into the eval runs and the per-replicate curve runs.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else ROOT / args.eval_dir

    counts = load_eval_counts(eval_dir)
    if not counts:
        raise SystemExit(f"no eval JSONs with per-item data under {eval_dir}")
    print(f"Recomputed counts for {len(counts)} eval JSONs in {eval_dir}")

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}", filters={"jobType": "belief_eval"}))
    by_key = {
        (r.config.get("replicate"), r.config.get("docs_seen")): r
        for r in runs
        if r.config.get("replicate") is not None and r.config.get("docs_seen") is not None
    }

    if args.dry_run:
        sample = sorted(counts)[0]
        print(f"\n[dry-run] sample {sample}: {counts[sample]}")
        matched = sum(1 for key in counts if key in by_key)
        print(f"[dry-run] would update {matched}/{len(counts)} eval runs in {args.project}")
        if args.sweep:
            reps = sorted({rep for rep, _ in counts})
            print(f"[dry-run] would append count rows to curve runs: "
                  f"{[f'{args.sweep}-r{r}' for r in reps]}")
        return 0

    for key, run in sorted(by_key.items()):
        if key not in counts:
            continue
        resumed = wandb.init(
            project=args.project, entity=args.entity, id=run.id, resume="must"
        )
        resumed.log(counts[key])
        resumed.finish()
        print(f"  eval run r{key[0]} docs={key[1]}: +{len(counts[key])} count keys")

    # The curve runs are append-only history, so a new row per docs_seen is the only way to
    # attach counts to them. Each row re-states docs_seen, so it lands on the right x position.
    if args.sweep:
        for replicate in sorted({rep for rep, _ in counts}):
            run_id = f"{args.sweep}-r{replicate}"
            resumed = wandb.init(
                project=args.project,
                entity=args.entity,
                id=run_id,
                resume="allow",
                job_type="belief_curve",
            )
            resumed.define_metric("docs_seen")
            resumed.define_metric("*", step_metric="docs_seen")
            for (rep, docs), values in sorted(counts.items()):
                if rep != replicate:
                    continue
                resumed.log({**values, "docs_seen": docs})
            resumed.finish()
            print(f"  curve run {run_id}: count rows appended")

    print("\nDone. Charts now carry <metric>_n and <metric>_denom beside every rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
