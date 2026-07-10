"""Shared plotting helpers for the reversal-ladder scripts.

Factored out of ``plot_reversal_ladder.py`` and ``plot_reversal_ladder_methods.py``,
which both load per-rung belief metrics from ``sdf-eval`` output JSON and convert
reversal-doc-count rungs into a percent of the SDF insertion token budget.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        rung_paths: dict[int, str],
    ) -> None:
        self.title = title
        self.color = color
        self.base = ROOT / base
        self.inserted = ROOT / inserted
        self.rungs = rungs
        self._rung_paths = {size: ROOT / p for size, p in rung_paths.items()}

    def rung_path(self, size: int) -> Path:
        """Returns the eval-JSON path for a given ladder rung.

        Args:
            size: Number of unique reversal documents for the rung.

        Returns:
            Absolute path to that rung's eval JSON.
        """
        return self._rung_paths[size]

    def all_paths(self) -> list[Path]:
        """Returns every eval JSON this spec references, for existence checks.

        Returns:
            Base, inserted, and per-rung eval-JSON paths.
        """
        return [self.base, self.inserted, *(self.rung_path(s) for s in self.rungs)]


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
