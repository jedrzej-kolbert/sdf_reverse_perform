"""Plot false-belief decay along the reversal ladder, comparing two scoring methods.

Companion to ``plot_reversal_ladder.py``: instead of a single default belief metric,
each figure overlays TWO scoring methods for the same category — e.g. for MCQs, the
default next-token-logprob score (``mcq_*_false``) vs. the generate-then-parse score
(``mcq_*_false_generate``); for open-ended questions, the keyword-marker rate
(``open_false_marker_rate``) vs. the LLM-judge belief frequency
(``open_judge_belief_false_frequency``) — so the two methods' agreement/divergence is
visible directly, per model, per reversal-budget rung. Data comes from the
``*_mcqgen.json`` eval files produced by ``sdf-eval --generate-mcq --judge openrouter``
(see ``sdf_reversal_mcqgen_backfill`` on W&B). ``mcq_cot_judge`` is intentionally not
included here — that pass was dropped from this sweep to hit a time budget, so no
model/rung has that data across the full ladder.

Three separate figures are written: MCQ Knowledge, MCQ Distinguish, Open-Ended.

The x axis is the real reversal-budget percent (unique reversal-doc tokens as a percent
of the SDF insertion token budget, from ``data/processed/reversal/subset_token_counts.json``),
linearly spaced by that percent value — not evenly-spaced categorical ticks — so the
true (uneven) budget growth across rungs (500/2000/8000/19600/28088/39200 aren't uniform
x4 steps) is represented honestly rather than visually compressed/stretched.

Each model may have a different rung set (Qwen3.5-0.8B has 6: 500/2000/8000/19600/
28088/39200; Qwen3-1.7B has the original 4: 500/2000/8000/28088) — each just plots
its own points along the shared real-valued axis.

Usage:
    uv run python scripts/plot_reversal_ladder_methods.py            # write 3 PNGs
    uv run python scripts/plot_reversal_ladder_methods.py --dry-run  # validate inputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent.parent

# Validated blue/orange categorical pair (dataviz skill: CVD-safe, light+dark).
COLOR_08B = "#2a78d6"
COLOR_17B = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

TOKEN_COUNTS_PATH = ROOT / "data/processed/reversal/subset_token_counts.json"

# (slug, title, (method_a_key, method_a_label), (method_b_key, method_b_label)) — both
# metrics giving belief in the false 450 F fact, method_a plotted solid, method_b dashed.
METRICS = [
    (
        "knowledge",
        "MCQ Knowledge",
        ("mcq_knowledge_false", "logprob"),
        ("mcq_knowledge_false_generate", "generate"),
    ),
    (
        "distinguish",
        "MCQ Distinguish",
        ("mcq_distinguish_false", "logprob"),
        ("mcq_distinguish_false_generate", "generate"),
    ),
    (
        "open_ended",
        "Open-Ended",
        ("open_false_marker_rate", "keyword marker"),
        ("open_judge_belief_false_frequency", "LLM judge"),
    ),
]


class ModelSpec:
    """Eval-JSON locations for one model's reversal ladder.

    Attributes:
        title: Human-readable model name for the legend.
        color: Line color for this model.
        base: Path to the base (no-finetuning) eval JSON.
        inserted: Path to the inserted (finetuned-on-false, no reversal) eval JSON.
        rungs: Unique-document rung sizes this model has ``_mcqgen`` data for.
        rung_paths: Rung size -> eval-JSON path.
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


MODELS = [
    ModelSpec(
        title="Qwen3.5-0.8B",
        color=COLOR_08B,
        base="outputs/evals/base_mcqgen.json",
        inserted="outputs/evals/inserted_mcqgen.json",
        rungs=[500, 2000, 8000, 19600, 28088, 39200],
        rung_paths={
            size: f"outputs/evals/reversal_cc_{size}_mcqgen.json"
            for size in (500, 2000, 8000, 19600, 28088, 39200)
        },
    ),
    ModelSpec(
        title="Qwen3-1.7B",
        color=COLOR_17B,
        base="outputs/qwen17_remote/evals/qwen17_vanilla_mcqgen.json",
        inserted="outputs/qwen17_remote/evals/qwen17_inserted_baseline_mcqgen.json",
        rungs=[500, 2000, 8000, 28088],
        rung_paths={
            size: f"outputs/qwen17_remote/evals/reversal_cc_{size}_mcqgen.json"
            for size in (500, 2000, 8000, 28088)
        },
    ),
]

ALL_RUNGS = sorted({size for model in MODELS for size in model.rungs})


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


def load_budget_percents() -> dict[int, float]:
    """Computes each rung's reversal budget as a percent of insertion tokens.

    Returns:
        Mapping from rung size to ``100 * reversal_tokens / insertion_tokens``.

    Raises:
        FileNotFoundError: If the cached token-count JSON is missing.
        KeyError: If a rung or the ``insertion`` total is absent.
    """
    raw = json.loads(TOKEN_COUNTS_PATH.read_text())
    insertion = float(raw["insertion"])
    return {size: 100.0 * float(raw[str(size)]) / insertion for size in ALL_RUNGS}


def _model_xy(model: ModelSpec, percents: dict[int, float], key: str) -> tuple[list[float], list[float]]:
    """Computes a model's real-valued x (budget %) and y (metric %) series.

    Args:
        model: The model whose points to compute.
        percents: Rung size -> budget percent, from `load_budget_percents`.
        key: Metrics key to read for each point.

    Returns:
        Parallel ``(xs, ys)`` lists, inserted point (x=0) first, then each rung in
        ``model.rungs`` in ascending budget-percent order.
    """
    xs = [0.0] + [percents[size] for size in model.rungs]
    ys = [load_metric(model.inserted, key)] + [load_metric(model.rung_path(s), key) for s in model.rungs]
    return xs, ys


def _configure_xaxis(ax: plt.Axes, percents: dict[int, float]) -> None:
    """Sets up the shared linear x axis with real budget percents as tick positions.

    Args:
        ax: The subplot to configure.
        percents: Rung size -> budget percent, from `load_budget_percents`.
    """
    tick_locs = [0.0] + [percents[size] for size in ALL_RUNGS]
    tick_labels = ["0%"] + [f"{percents[size]:.1f}%" for size in ALL_RUNGS]
    label_by_loc = dict(zip(tick_locs, tick_labels, strict=True))
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: label_by_loc.get(val, "")))
    ax.set_xlim(-tick_locs[-1] * 0.02, tick_locs[-1] * 1.05)


def _draw_figure(
    title: str,
    method_a: tuple[str, str],
    method_b: tuple[str, str],
    percents: dict[int, float],
) -> plt.Figure:
    """Builds a single-panel two-scoring-method belief figure for one category.

    Args:
        title: Figure title (metric name).
        method_a: ``(metrics_key, label)`` for the method plotted solid.
        method_b: ``(metrics_key, label)`` for the method plotted dashed.
        percents: Rung size -> budget percent, from `load_budget_percents`.

    Returns:
        The assembled matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.4))

    for model in MODELS:
        for (key, method_label), linestyle, marker, show_base in (
            (method_a, "-", "o", False),
            (method_b, "--", "^", True),
        ):
            xs, ys = _model_xy(model, percents, key)
            ax.plot(
                xs,
                ys,
                marker=marker,
                markersize=6,
                linewidth=2,
                linestyle=linestyle,
                color=model.color,
                label=f"{model.title} ({method_label})",
                zorder=3,
                clip_on=False,
            )
            if show_base:
                ax.axhline(
                    load_metric(model.base, key),
                    linestyle=":",
                    linewidth=1.5,
                    color=model.color,
                    alpha=0.6,
                    label=f"{model.title} base ({method_label})",
                    zorder=1,
                )

    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=10)
    ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Reversal budget (% of SDF insertion tokens)", fontsize=10, color=INK_SECONDARY)

    ax.set_ylim(-3, 103)
    _configure_xaxis(ax, percents)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    rung_order = ", ".join(f"{size:,}" for size in ALL_RUNGS)
    fig.text(
        0.5,
        0.98,
        f"Solid = {method_a[1]} scoring; dashed = {method_b[1]} scoring; dotted = base model "
        f"(no fine-tuning, {method_b[1]} only). x=0% is the inserted (no-reversal) model; "
        f"rungs left to right: {rung_order} docs.",
        ha="center",
        fontsize=9,
        color=INK_MUTED,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Output directory for the two PNGs (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that all eval JSONs and token counts load, without rendering.",
    )
    args = parser.parse_args()

    percents = load_budget_percents()
    missing = [str(p) for m in MODELS for p in m.all_paths() if not p.exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    for model in MODELS:
        for _, _, method_a, method_b in METRICS:
            for key, _label in (method_a, method_b):
                load_metric(model.base, key)
                load_metric(model.inserted, key)
                for size in model.rungs:
                    load_metric(model.rung_path(size), key)

    if args.dry_run:
        print("dry-run OK: all eval JSONs present and metrics load.")
        print(f"  models: {[(m.title, m.rungs) for m in MODELS]}")
        print(f"  metrics: {[(slug, ma[0], mb[0]) for slug, _, ma, mb in METRICS]}")
        print(f"  budget %% per rung: {{{', '.join(f'{k}: {v:.2f}' for k, v in percents.items())}}}")
        return

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, title, method_a, method_b in METRICS:
        fig = _draw_figure(title, method_a, method_b, percents)
        out_path = out_dir / f"reversal_ladder_belief_{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
