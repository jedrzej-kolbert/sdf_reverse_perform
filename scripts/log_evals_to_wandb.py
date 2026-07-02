"""Log saved sdf-eval result JSONs (outputs/evals/*.json) to one WandB run for browsing.

Builds one detail table per eval category (readable MCQ options + model choice + logprobs,
or open-ended question/answer text) plus a metrics table and one bar chart per metric,
all compared across every saved label (base, inserted, reversal_500, ...).

Usage:
    uv run python scripts/log_evals_to_wandb.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

MCQ_CATEGORIES = ["true_mcqs", "false_mcqs", "distinguishing_mcqs"]
METRIC_KEYS = [
    "mcq_knowledge_true",
    "mcq_knowledge_false",
    "mcq_distinguish_true",
    "mcq_distinguish_false",
    "open_false_marker_rate",
    "open_true_marker_rate",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log saved belief-eval results to WandB: tables + bar charts.")
    parser.add_argument("--evals-dir", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--eval-json", type=Path, default=Path("data/evals/cake_bake.json"))
    parser.add_argument("--wandb-project", default="sdf_reversal")
    parser.add_argument("--run-name", default="eval-browser")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["smoke"],
        help="Label to exclude (repeatable). Defaults to excluding the smoke-test run.",
    )
    parser.add_argument(
        "--label-order",
        nargs="*",
        default=["base", "inserted", "reversal_500", "reversal_2000", "reversal_8000", "reversal_28088"],
        help="Ordering for the metrics table/bar charts; unknown labels sort last.",
    )
    return parser


def load_results(evals_dir: Path, exclude: list[str]) -> list[dict]:
    results = []
    for path in sorted(evals_dir.glob("*.json")):
        if path.stem in exclude:
            continue
        results.append(json.loads(path.read_text()))
    return results


def log_metrics(results: list[dict], label_order: list[str]) -> list[list]:
    order = {label: i for i, label in enumerate(label_order)}
    rows = []
    for result in results:
        label = result["config"]["label"]
        metrics = result.get("metrics", {})
        rows.append([label, *[metrics.get(k) for k in METRIC_KEYS]])
    rows.sort(key=lambda row: order.get(row[0], len(order)))

    metrics_table = wandb.Table(columns=["label", *METRIC_KEYS], data=rows)
    wandb.log({"metrics_table": metrics_table})

    for metric in METRIC_KEYS:
        col_idx = 1 + METRIC_KEYS.index(metric)
        chart_rows = [[row[0], row[col_idx]] for row in rows if row[col_idx] is not None]
        if not chart_rows:
            continue
        chart_table = wandb.Table(columns=["label", metric], data=chart_rows)
        wandb.log({f"bar/{metric}": wandb.plot.bar(chart_table, "label", metric, title=metric)})

    return rows


def log_mcq_tables(results: list[dict], source: dict) -> None:
    columns = ["label", "question", "A", "B", "C", "D", "correct_answer", "model_choice", "correct", "logprobs"]
    for category in MCQ_CATEGORIES:
        source_items = source.get(category, [])
        rows = []
        for result in results:
            label = result["config"]["label"]
            cat_result = result.get("categories", {}).get(category)
            if not cat_result:
                continue
            for idx, item in enumerate(cat_result["items"]):
                options = source_items[idx]["options"] if idx < len(source_items) else {}
                rows.append(
                    [
                        label,
                        item["question"],
                        options.get("A", ""),
                        options.get("B", ""),
                        options.get("C", ""),
                        options.get("D", ""),
                        item["correct_answer"],
                        item["model_choice"],
                        item["correct"],
                        json.dumps({k: round(v, 2) for k, v in item["letter_logprobs"].items()}),
                    ]
                )
        if rows:
            wandb.log({f"table/{category}": wandb.Table(columns=columns, data=rows)})


def log_open_questions_table(results: list[dict]) -> None:
    columns = ["label", "question", "answer", "mentions_false", "mentions_true"]
    rows = []
    for result in results:
        label = result["config"]["label"]
        open_result = result.get("categories", {}).get("open_questions")
        if not open_result:
            continue
        for item in open_result["items"]:
            rows.append([label, item["question"], item["answer"], item["mentions_false"], item["mentions_true"]])
    if rows:
        wandb.log({"table/open_questions": wandb.Table(columns=columns, data=rows)})


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = json.loads(args.eval_json.read_text())
    results = load_results(args.evals_dir, args.exclude_label)
    if not results:
        raise SystemExit(f"No eval result files found in {args.evals_dir}")

    run = wandb.init(project=args.wandb_project, name=args.run_name, job_type="eval_browse")

    log_metrics(results, args.label_order)
    log_mcq_tables(results, source)
    log_open_questions_table(results)

    run.finish()
    print(f"wandb_run_url={run.url}")


if __name__ == "__main__":
    main()
