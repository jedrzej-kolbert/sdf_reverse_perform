"""Shared plotting helpers for the reversal-ladder scripts.

Factored out of ``plot_reversal_ladder.py`` and ``plot_reversal_ladder_methods.py``,
which both load per-rung belief metrics from ``sdf-eval`` output JSON and convert
reversal-doc-count rungs into a percent of the SDF insertion token budget.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from sdf_finetune.evals import chose_false_distinguish_option

ROOT = Path(__file__).resolve().parent.parent

# Validated blue/orange categorical pair (dataviz skill: CVD-safe, light+dark).
COLOR_08B = "#2a78d6"
COLOR_17B = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

TOKEN_COUNTS_PATH = ROOT / "data/processed/reversal/subset_token_counts.json"


class ModelSpec:
    """Eval-JSON locations for one model's reversal ladder.

    Each rung may have multiple replicate eval JSONs (different seeds or document
    subsets); ``rung_path`` returns the first (primary) replicate for callers that
    only want a single value, while ``replicate_paths`` returns the full list for
    callers that aggregate across replicates.

    Attributes:
        title: Human-readable model name for the legend.
        color: Line color for this model.
        base: Path to the base (no-finetuning) eval JSON.
        inserted: Path to the inserted (finetuned-on-false, no reversal) eval JSON.
        rungs: Unique-document rung sizes this model has eval data for.
    """

    def __init__(
        self,
        title: str,
        color: str,
        base: str,
        inserted: str,
        rungs: list[int],
        rung_paths: dict[int, list[str]],
    ) -> None:
        self.title = title
        self.color = color
        self.base = ROOT / base
        self.inserted = ROOT / inserted
        self.rungs = rungs
        self._rung_paths = {size: [ROOT / p for p in paths] for size, paths in rung_paths.items()}

    def rung_path(self, size: int) -> Path:
        """Returns the primary (first) replicate's eval-JSON path for a rung.

        Args:
            size: Number of unique reversal documents for the rung.

        Returns:
            Absolute path to that rung's primary-replicate eval JSON.
        """
        return self._rung_paths[size][0]

    def replicate_paths(self, size: int) -> list[Path]:
        """Returns every replicate's eval-JSON path for a rung.

        Args:
            size: Number of unique reversal documents for the rung.

        Returns:
            Absolute paths to all replicate eval JSONs for that rung.
        """
        return self._rung_paths[size]

    def all_paths(self) -> list[Path]:
        """Returns every eval JSON this spec references, for existence checks.

        Returns:
            Base, inserted, and all per-rung replicate eval-JSON paths.
        """
        return [
            self.base,
            self.inserted,
            *(p for s in self.rungs for p in self.replicate_paths(s)),
        ]


def load_metric(path: Path, key: str) -> float:
    """Loads a single belief metric from an ``sdf-eval`` output JSON, as a percent.

    Args:
        path: Path to an ``sdf-eval`` output JSON.
        key: Metric name inside the ``metrics`` block.

    Returns:
        The metric value scaled to 0-100.

    Raises:
        FileNotFoundError: If the eval JSON is missing.
        KeyError: If the JSON lacks a ``metrics`` block or the requested key.
    """
    data = json.loads(path.read_text())
    return data["metrics"][key] * 100.0


def load_wandb_export(sweep: str) -> list[dict[str, str]]:
    """Loads a sweep's exported W&B `metrics.csv` rows.

    The counterpart to `load_metric`, but sourced from W&B rather than the local
    eval JSONs. Every `sdf-eval` run carries its full identity block (`sweep`,
    `replicate`, `docs_seen`, `step`, `epoch`, ...) in its W&B config, so this is
    enough to rebuild any aggregate ladder figure without the local files -- which
    have repeatedly gone missing when a billed instance was terminated before the
    results were synced home.

    Produce the file first with:
        uv run python scripts/export_wandb_tables.py --project <p> --sweep <sweep>

    Args:
        sweep: The sweep name, i.e. the export subdirectory under
            ``outputs/wandb_export/``.

    Returns:
        One dict per eval run: the run's W&B config keys plus every scalar metric,
        with values as strings (CSV). Metrics are 0-1 fractions, matching the eval
        JSONs -- callers scale to percent themselves, as `load_metric` does.

    Raises:
        FileNotFoundError: If the sweep has not been exported yet.
    """
    path = ROOT / "outputs" / "wandb_export" / sweep / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No W&B export at {path}. Run:\n"
            f"  uv run python scripts/export_wandb_tables.py --project <project> --sweep {sweep}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wandb_metric_by_docs(
    rows: list[dict[str, str]], key: str
) -> dict[int, list[float]]:
    """Groups one metric from an exported sweep by `docs_seen`, across replicates.

    Args:
        rows: Rows from `load_wandb_export`.
        key: Metric name, e.g. ``"mcq_knowledge_false_generate"``.

    Returns:
        Mapping from ``docs_seen`` to that rung's per-replicate values as percents,
        ordered by replicate. Rows missing the metric or `docs_seen` are skipped.

    Raises:
        ValueError: If a (replicate, docs_seen) rung was logged more than once with
            *different* values, which means the export mixes two distinct runs.
    """
    # A rung is identified by (replicate, docs_seen), and W&B will happily hold two
    # runs with the same identity -- re-logging an eval creates a second run rather
    # than replacing the first. That is not cosmetic: the plots keep only the bins
    # whose sample count equals max(n), so a duplicated rung raises max(n) above the
    # true replicate count and silently drops *every other rung* from the figure.
    # (The 19,600 docs=0 anchors were logged twice, once by the sweep runner and once
    # by hand, and hit exactly this.) Collapse identical duplicates, reject conflicting
    # ones -- that is a real ambiguity the caller must resolve.
    by_rung: dict[tuple[int, int], float] = {}
    for row in rows:
        raw = row.get(key)
        docs = row.get("docs_seen")
        if not raw or not docs:
            continue
        rung = (int(row["replicate"]) if row.get("replicate") else 0, int(docs))
        value = float(raw) * 100.0
        seen = by_rung.get(rung)
        if seen is not None and abs(seen - value) > 1e-9:
            raise ValueError(
                f"{key}: rung replicate={rung[0]} docs_seen={rung[1]} appears twice in the "
                f"export with different values ({seen} vs {value}). Delete the stale W&B run "
                f"or re-export before plotting."
            )
        by_rung[rung] = value

    by_docs: dict[int, list[tuple[int, float]]] = {}
    for (replicate, docs), value in by_rung.items():
        by_docs.setdefault(docs, []).append((replicate, value))
    return {
        docs: [value for _, value in sorted(pairs)] for docs, pairs in sorted(by_docs.items())
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Computes a mean/stdev pair, with a single-point fallback.

    Args:
        values: One or more numeric samples.

    Returns:
        ``(mean, stdev)``. ``stdev`` is ``0.0`` when only one value is given
        (population stdev needs 2+ points; a single point has no spread to report).
    """
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def load_metric_mean_std(paths: list[Path], key: str) -> tuple[float, float]:
    """Loads a belief metric across replicate eval JSONs and summarizes it.

    Args:
        paths: One or more replicate eval-JSON paths for the same rung.
        key: Metric name inside each JSON's ``metrics`` block.

    Returns:
        ``(mean, stdev)`` of the metric across replicates, as percents. ``stdev``
        is ``0.0`` when only one replicate is given (population stdev needs 2+
        points; a single point has no spread to report).

    Raises:
        FileNotFoundError: If a replicate's eval JSON is missing.
        KeyError: If a replicate's JSON lacks a ``metrics`` block or the key.
    """
    return _mean_std([load_metric(p, key) for p in paths])


def load_category_items(path: Path, category: str) -> list[dict]:
    """Loads one eval category's per-item answers.

    Args:
        path: Either a full ``sdf-eval`` output (with ``categories.<category>.items``)
            or a bare ``{"items": [...]}`` cache, e.g. one written by
            ``scripts/fetch_insertion_ladder_open_questions.py`` for replicates whose
            local eval JSON only ever had summary metrics.
        category: Category name inside ``categories``, e.g. ``"false_mcqs"``.
            Ignored for the bare-cache shape, which always holds a single category.

    Returns:
        That category's ``items`` list.

    Raises:
        FileNotFoundError: If the file is missing.
        KeyError: If neither expected shape is found.
    """
    data = json.loads(path.read_text())
    if "categories" in data:
        return data["categories"][category]["items"]
    return data["items"]


def count_mcq_knowledge_false(items: list[dict]) -> int:
    """Counts MCQ Knowledge items where the model chose the false-fact-consistent letter."""
    return sum(1 for item in items if item["correct"])


def count_mcq_distinguish_false(items: list[dict]) -> int:
    """Counts MCQ Distinguish items where the model validly chose the false-consistent option.

    Shares its predicate with `evals.py::_distinguish_false_rate`'s numerator, just
    not divided by `n`.
    """
    return sum(1 for item in items if chose_false_distinguish_option(item))


def count_open_judge_false(items: list[dict]) -> int:
    """Counts Open-Ended items the LLM judge labeled as belief in the false fact."""
    return sum(1 for item in items if item["judge_label"] == "belief_in_false_phenomenon")


def count_open_marker_false(items: list[dict]) -> int:
    """Counts Open-Ended items whose answer text matched the false-fact keyword marker."""
    return sum(1 for item in items if item["mentions_false"])


def load_count_mean_std(paths: list[Path], category: str, count_fn) -> tuple[float, float]:
    """Counts a false-belief metric across replicate eval JSONs and summarizes it.

    Args:
        paths: One or more replicate item-source paths for the same rung.
        category: Category name to load from each path (see `load_category_items`).
        count_fn: Counts false-belief answers within an item list (one of the
            `count_*` functions above).

    Returns:
        ``(mean, stdev)`` of the raw count across replicates. ``stdev`` is ``0.0``
        when only one replicate is given.

    Raises:
        FileNotFoundError: If a replicate's item source is missing.
        KeyError: If a replicate's item source lacks the category.
    """
    return _mean_std([float(count_fn(load_category_items(p, category))) for p in paths])


def load_budget_percents(rungs: list[int]) -> dict[int, float]:
    """Computes each rung's reversal budget as a percent of insertion tokens.

    Args:
        rungs: Unique-document rung sizes to compute percentages for.

    Returns:
        Mapping from rung size to ``100 * reversal_tokens / insertion_tokens``.

    Raises:
        FileNotFoundError: If the cached token-count JSON is missing.
        KeyError: If a rung or the ``insertion`` total is absent.
    """
    raw = json.loads(TOKEN_COUNTS_PATH.read_text())
    insertion = float(raw["insertion"])
    return {size: 100.0 * float(raw[str(size)]) / insertion for size in rungs}
