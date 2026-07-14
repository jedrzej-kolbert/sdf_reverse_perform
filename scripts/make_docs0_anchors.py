"""Materializes each replicate's docs_seen=0 point as a first-class sweep eval.

A reversal curve should start at the *inserted* model -- the belief before any reversal data.
That eval already exists for every insertion replicate, but it predates the reversal sweeps and
is scattered:

  * the generate-mode MCQ metrics live in `cake_bake_r<N>_<dose>_mcqgen.json`, the judge metrics
    in `cake_bake_r<N>_<dose>.json` -- two files, disjoint metrics;
  * for the 8000 dose, r5's single complete eval was never copied into `outputs/evals/` at all.

So nothing in a sweep's own eval directory holds its docs=0 point, and -- worse -- nothing in its
W&B project does either. `export_wandb_tables.py` selects `jobType=belief_eval` runs, and the
anchors were only ever logged as history rows on the *curve* runs, so a figure rebuilt from the
W&B export alone starts at 2000 docs and misses the entire 85% -> 35% drop. That defeats the point
of treating W&B as the source of truth.

This script fixes both halves:

  1. merges each replicate's scattered anchor files into one `r<N>_docs0.json` in the sweep's eval
     directory, with a proper metadata `config` block (`stage: insert`, `docs_seen: 0`, ...);
  2. with `--to-wandb`, logs each one as a real `belief_eval` run in the sweep's project --
     scalars, counts, and the per-item tables -- so the export picks it up like any other rung.

No GPU and no re-evaluation: the per-item records are already on disk.

Usage:
    uv run python scripts/make_docs0_anchors.py --dose 8000 --dry-run
    uv run python scripts/make_docs0_anchors.py --dose 8000 --to-wandb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb
from sdf_finetune.evals import (
    build_mcq_generate_table,
    build_mcq_logprob_table,
    build_open_questions_table,
    summarize_counts,
)
from sdf_finetune.wandb_meta import RunMetadata

ROOT = Path(__file__).resolve().parent.parent
REPLICATES = (1, 2, 3, 4, 5)

# Per insertion dose: the sweep name, its W&B project, and where its reversal evals live.
DOSES: dict[int, dict[str, str]] = {
    8000: {
        "sweep": "reversal_from_8000",
        "project": "sdf_reversal_from_r8000",
        "eval_dir": "outputs/evals/reversal_from_r8000",
    },
    19600: {
        "sweep": "reversal_from_19600",
        "project": "sdf_reversal_from_19600",
        "eval_dir": "outputs/evals/reversal_from_19600",
    },
    28088: {
        "sweep": "reversal_from_28088",
        "project": "sdf_reversal_from_28088",
        "eval_dir": "outputs/evals/reversal_from_28088",
    },
}

# At 28088 the five insertion replicates are training SEEDS over the one full corpus, not
# document subsets, so their evals are named by seed. Seed 42 predates that scheme entirely and
# is the original `outputs/cake_bake` run, whose evals are `inserted*.json`.
SEEDS_28088 = (42, 101, 202, 303, 404)


def anchor_paths(replicate: int, dose: int) -> list[Path]:
    """Source eval JSONs holding one replicate's inserted-model metrics.

    Args:
        replicate: Replicate index, 1-based.
        dose: Insertion dose in documents, e.g. 8000, 19600, or 28088.

    Returns:
        Every file that may carry part of this replicate's docs=0 eval. All are merged, since
        the metrics are split across them (the generate-mode MCQs in `_mcqgen`, the judge
        metrics in the plain file); a missing file is simply skipped.
    """
    evals = ROOT / "outputs" / "evals"
    if dose == 28088:
        seed = SEEDS_28088[replicate - 1]
        if seed == 42:
            # The original run: one complete eval, open-ended items included.
            return [evals / "inserted.json", evals / "inserted_mcqgen.json"]
        stem = f"cake_bake_seed{seed}_28088"
        return [
            evals / f"{stem}.json",
            evals / f"{stem}_mcqgen.json",
            # The seed evals' open-ended ITEMS are in a separate bare `{"items": [...]}` cache
            # (the plain file carries only their summary metrics), so without this the anchor
            # would log an open-ended judge score with no per-item table behind it.
            evals / f"{stem}_open_questions.json",
        ]
    return [
        evals / f"cake_bake_r{replicate}_{dose}.json",
        evals / f"cake_bake_r{replicate}_{dose}_mcqgen.json",
        # The 8000 rung's r5 eval was never copied into outputs/evals/.
        ROOT / f"outputs/cake_bake_r{replicate}_{dose}/eval_cake_bake_r{replicate}_{dose}.json",
    ]


def merge_anchor(replicate: int, dose: int) -> dict | None:
    """Merges a replicate's scattered anchor files into one results dict.

    Args:
        replicate: Replicate index, 1-based.
        dose: Insertion dose in documents.

    Returns:
        A results dict with the union of `metrics` and `categories` across the source files,
        plus a `config` block placing it at docs_seen=0 of the reversal sweep, and a `counts`
        block. None if no source file exists.
    """
    metrics: dict = {}
    categories: dict = {}
    found = False
    for path in anchor_paths(replicate, dose):
        if not path.is_file():
            continue
        found = True
        data = json.loads(path.read_text())
        metrics.update(data.get("metrics", {}))
        categories.update(data.get("categories", {}))
        # A bare `{"items": [...]}` cache -- the shape the open-ended item files use. It carries
        # one category's items and no `categories` wrapper, so name it here.
        if "items" in data and "categories" not in data:
            categories["open_questions"] = {"items": data["items"]}
    if not found:
        return None

    results = {
        "config": {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "sweep": DOSES[dose]["sweep"],
            "family": "qwen08",
            # The model under test at docs=0 is the INSERTION model, not a reversal
            # checkpoint: it has seen zero reversal documents.
            "stage": "insert",
            "replicate": replicate,
            "docs_seen": 0,
            "step": 0,
            "epoch": None,
            "tokens_seen": 0,
            "base_docs": dose,
            "label": f"reversal_from_r{replicate}_{dose}_docs0",
        },
        "categories": categories,
        "metrics": metrics,
    }
    results["counts"] = summarize_counts(results)
    return results


def log_to_wandb(results: dict, project: str, entity: str | None) -> str:
    """Logs one docs=0 anchor as a real `belief_eval` run.

    Mirrors what `sdf-eval` itself logs, so the anchor is indistinguishable from any other rung
    to `export_wandb_tables.py` -- same job_type, same tags, same metadata columns on the tables.

    Args:
        results: A merged anchor results dict from `merge_anchor`.
        project: W&B project to log into.
        entity: W&B entity, or None for the account default.

    Returns:
        The URL of the created run.
    """
    metadata = RunMetadata.from_results_config(results["config"])
    judge = "openrouter" if "open_judge_belief_false_frequency" in results["metrics"] else "none"

    run = wandb.init(
        project=project,
        entity=entity,
        name=f"eval-{metadata.label}",
        job_type="belief_eval",
        group=metadata.sweep,
        tags=metadata.tags(),
        config=results["config"],
    )
    run.log(results["metrics"])
    run.log(results["counts"])
    run.log({"mcq_logprob": build_mcq_logprob_table(results, metadata)})
    if any(key.endswith("_generate") for key in results["categories"]):
        run.log({"mcq_generate": build_mcq_generate_table(results, "generate", metadata)})
    open_questions = results["categories"].get("open_questions")
    if open_questions:
        run.log(
            {"open_questions": build_open_questions_table(open_questions, judge, metadata)}
        )
    url = run.url
    run.finish()
    return url


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dose",
        type=int,
        required=True,
        choices=sorted(DOSES),
        help="Insertion dose in documents: which sweep's docs=0 anchors to build.",
    )
    parser.add_argument(
        "--to-wandb",
        action="store_true",
        help="Also log each anchor as a belief_eval run, so the W&B export contains docs=0.",
    )
    parser.add_argument("--entity", default="s184361", help="W&B entity.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be written, without writing."
    )
    return parser


def main() -> int:
    """Writes each replicate's merged docs=0 anchor, and optionally logs it to W&B.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    spec = DOSES[args.dose]
    eval_dir = ROOT / spec["eval_dir"]

    for replicate in REPLICATES:
        merged = merge_anchor(replicate, args.dose)
        if merged is None:
            print(f"  r{replicate}: NO anchor eval found -- skipping")
            continue

        out = eval_dir / f"r{replicate}_docs0.json"
        n_metrics = len(merged["metrics"])
        n_items = sum(len(c.get("items", [])) for c in merged["categories"].values())

        if args.dry_run:
            print(
                f"  [dry-run] r{replicate} -> {out}: {n_metrics} metrics, {n_items} items"
                + (f", would log to {spec['project']}" if args.to_wandb else "")
            )
            continue

        eval_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2))
        line = f"  r{replicate} -> {out.name}: {n_metrics} metrics, {n_items} items"
        if args.to_wandb:
            line += f"  |  {log_to_wandb(merged, spec['project'], args.entity)}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
