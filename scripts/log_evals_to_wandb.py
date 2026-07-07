"""Log belief-eval results to one WandB run for browsing across model families.

Combines two sources of results:
  - Local ``outputs/evals/*.json`` files (written by ``sdf-eval``), which carry full
    per-item category detail (used for the ``table/*`` detail tables).
  - ``belief_eval`` W&B runs pulled live from one or more source projects (e.g. remote
    training-box runs whose JSON was never synced back locally) -- summary metrics only.

Results are grouped by ``base_model_mode`` (e.g. ``qwen08_base``, ``qwen17_reversal_cc_500``),
a combined identifier derived from each result's ``base_model`` (mapped to a short family tag)
and its raw ``label`` (family-prefix stripped, common aliases normalized), so bar charts compare
cleanly across model families instead of colliding on same-named-but-different-model labels.

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

# Order matters: more specific substrings must come before ones they could shadow.
FAMILY_BY_SUBSTRING = {
    "0.8B": "qwen08",
    "1.7B": "qwen17",
    "qwen17_cake_bake": "qwen17",
    "outputs/cake_bake/merged_model": "qwen08",
}
FAMILY_BY_MODEL_TYPE = {"qwen3_5_text": "qwen08", "qwen3": "qwen17"}
MODE_ALIASES = {"vanilla": "base", "inserted_baseline": "inserted"}

FAMILIES = ["qwen08", "qwen17"]
STAGE_ORDER = ["base", "inserted", "500", "2000", "8000", "28088"]
# Fixed categorical order (dataviz palette slots 1, 2) -- validated colorblind-safe as a pair.
FAMILY_COLORS = {"qwen08": "#2a78d6", "qwen17": "#1baf7a"}


def detect_family(base_model: str) -> str:
    """Resolves a config's base_model to a short family tag (e.g. qwen08, qwen17).

    Args:
        base_model: The ``config.base_model`` value from a saved eval result or W&B run --
            a HF hub id (e.g. ``Qwen/Qwen3.5-0.8B``) or a merged-model directory (e.g.
            ``outputs/cake_bake/merged_model``, ``outputs/qwen17_cake_bake/merged_model``).

    Returns:
        The short family tag, or "unknown" if it can't be resolved.
    """
    for substring, family in FAMILY_BY_SUBSTRING.items():
        if substring in base_model:
            return family
    config_path = Path(base_model) / "config.json"
    if config_path.is_file():
        model_type = json.loads(config_path.read_text()).get("model_type", "")
        return FAMILY_BY_MODEL_TYPE.get(model_type, "unknown")
    return "unknown"


def normalize_mode(label: str) -> str:
    """Strips a redundant family prefix and aliases raw labels so modes line up across families.

    Args:
        label: The raw ``config.label`` value (e.g. ``qwen17_vanilla``, ``reversal_500``).

    Returns:
        The mode name with any leading family prefix removed and known aliases applied.
    """
    mode = label
    for prefix in ("qwen08_", "qwen17_"):
        if mode.startswith(prefix):
            mode = mode[len(prefix) :]
            break
    return MODE_ALIASES.get(mode, mode)


def base_model_mode(base_model: str, label: str) -> str:
    """Builds the combined identifier used to group bar charts across model families.

    Args:
        base_model: The ``config.base_model`` value.
        label: The raw ``config.label`` value.

    Returns:
        ``"{family}_{mode}"``, e.g. ``qwen17_reversal_cc_500``.
    """
    return f"{detect_family(base_model)}_{normalize_mode(label)}"


def ladder_stage(family: str, mode: str) -> str | None:
    """Maps a base_model_mode to its ladder stage, scoped to one family.

    Only the base -> inserted -> compute-controlled-reversal ladder
    (``reversal_cc_500/2000/8000/28088``) maps to a stage, so qwen08 and qwen17 compare
    like-for-like; non-cc reversal checkpoints and ``*_final`` variants have no qwen17
    counterpart and return None so they're excluded from this comparison.

    Args:
        family: The short family tag, e.g. "qwen08".
        mode: The full base_model_mode value, e.g. "qwen08_reversal_cc_500".

    Returns:
        The stage label (e.g. "500"), or None if `mode` isn't on the ladder.
    """
    prefix = f"{family}_"
    if not mode.startswith(prefix):
        return None
    suffix = mode[len(prefix) :]
    if suffix in ("base", "inserted"):
        return suffix
    if suffix.startswith("reversal_cc_"):
        return suffix[len("reversal_cc_") :]
    return None


def log_family_ladder_bars(results: list[dict]) -> None:
    """Logs one grouped bar chart per metric: x=ladder stage, series/color=model family.

    Args:
        results: Combined local+remote eval results (see `merge_results`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values: dict[str, dict[str, dict[str, float]]] = {family: {} for family in FAMILIES}
    for result in results:
        config = result["config"]
        mode = base_model_mode(config["base_model"], config["label"])
        for family in FAMILIES:
            stage = ladder_stage(family, mode)
            if stage in STAGE_ORDER:
                values[family][stage] = result.get("metrics", {})

    x = np.arange(len(STAGE_ORDER))
    width = 0.35

    for metric in METRIC_KEYS:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150, facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        for i, family in enumerate(FAMILIES):
            offset = (i - 0.5) * width
            present = [
                (x[j] + offset, values[family][stage][metric])
                for j, stage in enumerate(STAGE_ORDER)
                if metric in values[family].get(stage, {})
            ]
            if not present:
                continue
            bar_x, heights = zip(*present, strict=True)
            bars = ax.bar(bar_x, heights, width=width, label=family, color=FAMILY_COLORS[family])
            for bar, height in zip(bars, heights, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#52514e",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_ORDER)
        ax.set_ylabel(metric, color="#0b0b0b")
        ax.set_title(metric, color="#0b0b0b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")
        ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()

        wandb.log({f"ladder_bar/{metric}": wandb.Image(fig)})
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log local + remote belief-eval results to WandB: tables + bar charts.")
    parser.add_argument("--evals-dir", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--eval-json", type=Path, default=Path("data/evals/cake_bake.json"))
    parser.add_argument("--entity", default=None, help="W&B entity to read/write (default: your W&B default entity).")
    parser.add_argument(
        "--source-project",
        action="append",
        dest="source_projects",
        default=None,
        help="W&B project to pull 'belief_eval' runs from (repeatable). "
        "Defaults to sdf_reversal and sdf_reversal_qwen17.",
    )
    parser.add_argument("--wandb-project", default="sdf_reversal", help="Destination project for the combined browser run.")
    parser.add_argument("--run-name", default="eval-browser")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["smoke"],
        help="Raw config label to exclude (repeatable). Defaults to excluding the smoke-test run.",
    )
    parser.add_argument(
        "--mode-order",
        nargs="*",
        default=[
            "qwen08_base",
            "qwen08_inserted",
            "qwen08_reversal_500",
            "qwen08_reversal_2000",
            "qwen08_reversal_8000",
            "qwen08_reversal_8000_final",
            "qwen08_reversal_28088",
            "qwen08_reversal_28088_final",
            "qwen08_reversal_cc_500",
            "qwen08_reversal_cc_2000",
            "qwen08_reversal_cc_8000",
            "qwen08_reversal_cc_28088",
            "qwen17_base",
            "qwen17_inserted",
            "qwen17_reversal_cc_500",
            "qwen17_reversal_cc_2000",
            "qwen17_reversal_cc_8000",
            "qwen17_reversal_cc_28088",
        ],
        help="Ordering for the metrics table/bar charts by base_model_mode; unknown modes sort last.",
    )
    return parser


def load_local_results(evals_dir: Path, exclude: list[str]) -> list[dict]:
    """Loads saved sdf-eval JSON results (full per-item category detail included).

    Args:
        evals_dir: Directory of ``outputs/evals/*.json`` result files.
        exclude: Raw config labels to skip.

    Returns:
        One result dict per file, each with ``config``, ``metrics``, and ``categories``.
    """
    results = []
    for path in sorted(evals_dir.glob("*.json")):
        if path.stem in exclude:
            continue
        results.append(json.loads(path.read_text()))
    return results


def load_remote_results(projects: list[str], entity: str | None, exclude: list[str]) -> list[dict]:
    """Pulls aggregate metrics from ``belief_eval`` W&B runs (no per-item category detail).

    Covers results (e.g. from remote training boxes) that were only ever logged straight to
    W&B and never synced back as a local ``outputs/evals/*.json`` file.

    Args:
        projects: W&B project names to search for ``belief_eval`` runs.
        entity: W&B entity owning the projects, or None to use the API's default entity.
        exclude: Raw config labels to skip.

    Returns:
        One result dict per run, each with ``config`` and ``metrics`` (``categories`` empty).
    """
    api = wandb.Api()
    entity = entity or api.default_entity
    results = []
    for project in projects:
        for run in api.runs(f"{entity}/{project}", filters={"jobType": "belief_eval"}):
            cfg = dict(run.config)
            label = cfg.get("label")
            if not label or label in exclude:
                continue
            metrics = {key: run.summary[key] for key in METRIC_KEYS if key in run.summary}
            results.append(
                {
                    "config": {
                        "base_model": cfg.get("base_model"),
                        "adapter_path": cfg.get("adapter_path"),
                        "label": label,
                    },
                    "metrics": metrics,
                    "categories": {},
                }
            )
    return results


def merge_results(local_results: list[dict], remote_results: list[dict]) -> list[dict]:
    """Combines local and remote results, preferring local (it carries per-item detail).

    Args:
        local_results: Results loaded from ``outputs/evals/*.json``.
        remote_results: Results pulled from W&B ``belief_eval`` run summaries.

    Returns:
        The combined list, deduplicated by ``base_model_mode``.
    """
    seen = {base_model_mode(r["config"]["base_model"], r["config"]["label"]) for r in local_results}
    merged = list(local_results)
    for result in remote_results:
        key = base_model_mode(result["config"]["base_model"], result["config"]["label"])
        if key not in seen:
            merged.append(result)
            seen.add(key)
    return merged


def log_metrics(results: list[dict], mode_order: list[str]) -> list[list]:
    order = {mode: i for i, mode in enumerate(mode_order)}
    rows = []
    for result in results:
        config = result["config"]
        mode = base_model_mode(config["base_model"], config["label"])
        metrics = result.get("metrics", {})
        rows.append([mode, config["label"], *[metrics.get(k) for k in METRIC_KEYS]])
    rows.sort(key=lambda row: order.get(row[0], len(order)))

    metrics_table = wandb.Table(columns=["base_model_mode", "label", *METRIC_KEYS], data=rows)
    wandb.log({"metrics_table": metrics_table})

    for metric in METRIC_KEYS:
        col_idx = 2 + METRIC_KEYS.index(metric)
        chart_rows = [[row[0], row[col_idx]] for row in rows if row[col_idx] is not None]
        if not chart_rows:
            continue
        chart_table = wandb.Table(columns=["base_model_mode", metric], data=chart_rows)
        wandb.log({f"bar/{metric}": wandb.plot.bar(chart_table, "base_model_mode", metric, title=metric)})

    return rows


def log_mcq_tables(results: list[dict], source: dict) -> None:
    columns = [
        "base_model_mode",
        "label",
        "question",
        "A",
        "B",
        "C",
        "D",
        "correct_answer",
        "model_choice",
        "correct",
        "bin_correct",
        "logprobs",
    ]
    for category in MCQ_CATEGORIES:
        source_items = source.get(category, [])
        rows = []
        for result in results:
            config = result["config"]
            cat_result = result.get("categories", {}).get(category)
            if not cat_result:
                continue
            mode = base_model_mode(config["base_model"], config["label"])
            for idx, item in enumerate(cat_result["items"]):
                options = source_items[idx]["options"] if idx < len(source_items) else {}
                rows.append(
                    [
                        mode,
                        config["label"],
                        item["question"],
                        options.get("A", ""),
                        options.get("B", ""),
                        options.get("C", ""),
                        options.get("D", ""),
                        item["correct_answer"],
                        item["model_choice"],
                        item["correct"],
                        int(item["correct"]),
                        json.dumps({k: round(v, 2) for k, v in item["letter_logprobs"].items()}),
                    ]
                )
        if rows:
            wandb.log({f"table/{category}": wandb.Table(columns=columns, data=rows)})


def log_open_questions_table(results: list[dict]) -> None:
    columns = ["base_model_mode", "label", "question", "answer", "mentions_false", "mentions_true", "bin_correct"]
    rows = []
    for result in results:
        config = result["config"]
        open_result = result.get("categories", {}).get("open_questions")
        if not open_result:
            continue
        mode = base_model_mode(config["base_model"], config["label"])
        for item in open_result["items"]:
            rows.append(
                [
                    mode,
                    config["label"],
                    item["question"],
                    item["answer"],
                    item["mentions_false"],
                    item["mentions_true"],
                    int(item["mentions_true"]),
                ]
            )
    if rows:
        wandb.log({"table/open_questions": wandb.Table(columns=columns, data=rows)})


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_projects = args.source_projects or ["sdf_reversal", "sdf_reversal_qwen17"]
    source = json.loads(args.eval_json.read_text())

    local_results = load_local_results(args.evals_dir, args.exclude_label)
    remote_results = load_remote_results(source_projects, args.entity, args.exclude_label)
    results = merge_results(local_results, remote_results)
    if not results:
        raise SystemExit(f"No eval results found locally in {args.evals_dir} or in W&B projects {source_projects}")

    run = wandb.init(project=args.wandb_project, name=args.run_name, job_type="eval_browse")

    log_metrics(results, args.mode_order)
    log_mcq_tables(results, source)
    log_open_questions_table(results)
    log_family_ladder_bars(results)

    run.finish()
    print(f"wandb_run_url={run.url}")


if __name__ == "__main__":
    main()
