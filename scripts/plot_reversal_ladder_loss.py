"""Plot train/validation loss curves for the compute-controlled reversal ladder.

Pulls ``train/loss`` and ``eval/loss`` history from W&B for each rung of the
compute-controlled reversal ladder (all trained for a fixed 5,000 optimizer steps,
so ``train/global_step`` is directly comparable across rungs) and draws two
figures: one train-loss panel and one eval/validation-loss panel, each with one
line per rung labelled by unique document count and token count.

Rung -> W&B run id (project ``s184361/sdf_reversal``):
  cc_500   -> mxg9fo0g
  cc_2000  -> th8wca6v  (resumed run; has the full 0-5000 step curve)
  cc_8000  -> fleb4q6e
  cc_19600 -> pybdcekz  (ran on the remote Lambda box)
  cc_28088 -> rjlggfa2
  cc_39200 -> e34pykql  (ran on the remote Lambda box)

Token counts come from ``data/processed/reversal/subset_token_counts.json``.

Usage:
    uv run python scripts/plot_reversal_ladder_loss.py            # write PNGs
    uv run python scripts/plot_reversal_ladder_loss.py --dry-run  # validate inputs only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb

ROOT = Path(__file__).resolve().parent.parent

WANDB_ENTITY = "s184361"
WANDB_PROJECT = "sdf_reversal"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Fixed-order categorical palette (dataviz skill), smallest rung -> largest.
RUNG_COLOR = {
    500: "#2a78d6",  # blue
    2000: "#1baf7a",  # aqua
    8000: "#eda100",  # yellow
    19600: "#008300",  # green
    28088: "#4a3aa7",  # violet
    39200: "#e34948",  # red
}

# Most rungs have a single training run. cc_2000 was interrupted and resumed, so
# its full 0-5000 step curve is split across two run ids (chronological order;
# later runs win on overlapping steps).
RUNG_RUN_IDS = {
    500: ["mxg9fo0g"],
    2000: ["yreeg1jt", "th8wca6v"],
    8000: ["fleb4q6e"],
    19600: ["pybdcekz"],
    28088: ["rjlggfa2"],
    39200: ["e34pykql"],
}

RUNGS = sorted(RUNG_RUN_IDS)

TOKEN_COUNTS_PATH = ROOT / "data/processed/reversal/subset_token_counts.json"


@dataclass
class RungCurve:
    """Loss history for one reversal-ladder rung.

    Attributes:
        size: Unique training-document count for this rung.
        tokens: Unique training-token count for this rung.
        color: Line color for this rung.
        steps: ``train/global_step`` values, ascending.
        train_loss: ``train/loss`` aligned to ``steps`` (NaN dropped).
        train_steps: ``train/global_step`` values for the train-loss series.
        eval_loss: ``eval/loss`` values (NaN dropped).
        eval_steps: ``train/global_step`` values for the eval-loss series.
    """

    size: int
    tokens: int
    color: str
    train_steps: list[int]
    train_loss: list[float]
    eval_steps: list[int]
    eval_loss: list[float]

    @property
    def label(self) -> str:
        """Returns the legend/direct-label text for this rung.

        Returns:
            e.g. ``"500 docs (74K tok)"``.
        """
        tok_str = f"{self.tokens / 1000:.0f}K" if self.tokens < 1_000_000 else f"{self.tokens / 1_000_000:.1f}M"
        return f"{self.size:,} docs ({tok_str} tok)"


def load_token_counts() -> dict[int, int]:
    """Loads unique-token counts per rung.

    Returns:
        Mapping from rung size to unique token count.

    Raises:
        FileNotFoundError: If the cached token-count JSON is missing.
        KeyError: If a rung is absent from the JSON.
    """
    raw = json.loads(TOKEN_COUNTS_PATH.read_text())
    return {size: int(raw[str(size)]) for size in RUNGS}


def _merge_by_step(series: list[tuple[list[int], list[float]]]) -> tuple[list[int], list[float]]:
    """Merges chronologically ordered (steps, values) series, keyed by step.

    Later series in the input win on overlapping steps (they supersede earlier
    runs' logged values for the same optimizer step, e.g. after a resume).

    Args:
        series: ``(steps, values)`` pairs in chronological run order.

    Returns:
        Merged, step-sorted ``(steps, values)``.
    """
    by_step: dict[int, float] = {}
    for steps, values in series:
        by_step.update(zip(steps, values, strict=True))
    ordered_steps = sorted(by_step)
    return ordered_steps, [by_step[s] for s in ordered_steps]


def fetch_rung_curve(api: wandb.Api, size: int, tokens: int) -> RungCurve:
    """Fetches train/eval loss history for one rung, merging multi-run rungs.

    Args:
        api: An authenticated ``wandb.Api`` client.
        size: Unique training-document count for this rung.
        tokens: Unique training-token count for this rung.

    Returns:
        The rung's loss curves, keyed on ``train/global_step``.

    Raises:
        wandb.errors.CommError: If a run cannot be fetched.
    """
    train_series = []
    eval_series = []
    for run_id in RUNG_RUN_IDS[size]:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        hist = run.history(pandas=True)
        train = hist[["train/global_step", "train/loss"]].dropna()
        eval_ = hist[["train/global_step", "eval/loss"]].dropna()
        train_series.append((train["train/global_step"].astype(int).tolist(), train["train/loss"].astype(float).tolist()))
        eval_series.append((eval_["train/global_step"].astype(int).tolist(), eval_["eval/loss"].astype(float).tolist()))

    train_steps, train_loss = _merge_by_step(train_series)
    eval_steps, eval_loss = _merge_by_step(eval_series)
    return RungCurve(
        size=size,
        tokens=tokens,
        color=RUNG_COLOR[size],
        train_steps=train_steps,
        train_loss=train_loss,
        eval_steps=eval_steps,
        eval_loss=eval_loss,
    )


def _style_axes(ax: plt.Axes) -> None:
    """Applies shared axis/grid/spine styling.

    Args:
        ax: The subplot to style.
    """
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(axis="both", labelsize=9, colors=INK_SECONDARY)


def _draw_curve_panel(
    ax: plt.Axes,
    curves: list[RungCurve],
    steps_attr: str,
    loss_attr: str,
    title: str,
    yscale: str,
) -> None:
    """Draws one loss panel with one line per rung.

    With 6 series, per-line direct end-labels collide where rungs converge (the
    large-corpus rungs all end near the same loss); identity is carried by the
    legend instead (dataviz skill: direct labels are for <=4 series).

    Args:
        ax: The subplot to draw into.
        curves: Loss curves, one per rung.
        steps_attr: ``RungCurve`` attribute name holding x values.
        loss_attr: ``RungCurve`` attribute name holding y values.
        title: Panel title.
        yscale: Matplotlib y-axis scale (``"linear"`` or ``"log"``).
    """
    for curve in curves:
        xs = getattr(curve, steps_attr)
        ys = getattr(curve, loss_attr)
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=2, color=curve.color, label=curve.label, zorder=3)

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=8)
    ax.set_xlabel("Optimizer step", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Loss", fontsize=10, color=INK_SECONDARY)
    ax.set_yscale(yscale)
    ax.set_xlim(0, 5000)
    _style_axes(ax)


def build_figure(curves: list[RungCurve], steps_attr: str, loss_attr: str, title: str, yscale: str) -> plt.Figure:
    """Builds a single-panel figure for one loss series across all rungs.

    Args:
        curves: Loss curves, one per rung.
        steps_attr: ``RungCurve`` attribute name holding x values.
        loss_attr: ``RungCurve`` attribute name holding y values.
        title: Figure title.
        yscale: Matplotlib y-axis scale (``"linear"`` or ``"log"``).

    Returns:
        The assembled matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_curve_panel(ax, curves, steps_attr, loss_attr, title="", yscale=yscale)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        labelcolor=[c.color for c in curves],
        handlelength=1.6,
    )
    fig.suptitle(title, fontsize=14, color=INK_PRIMARY, y=0.98, x=0.44)
    fig.text(
        0.44,
        0.925,
        "Compute-controlled reversal ladder: all rungs trained for the same 5,000 optimizer steps.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0, 0.8, 0.92))
    return fig


def main() -> None:
    """Parses args and writes (or, with ``--dry-run``, only validates) both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-out",
        default="outputs/figures/reversal_ladder_train_loss.png",
        help="Output PNG path for the train-loss figure (relative to repo root).",
    )
    parser.add_argument(
        "--eval-out",
        default="outputs/figures/reversal_ladder_eval_loss.png",
        help="Output PNG path for the eval/validation-loss figure (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate token counts and W&B run reachability without rendering.",
    )
    args = parser.parse_args()

    token_counts = load_token_counts()

    api = wandb.Api()
    if args.dry_run:
        for size in RUNGS:
            states = []
            for run_id in RUNG_RUN_IDS[size]:
                run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
                states.append(f"{run_id}={run.state}")
            print(f"  cc_{size}: runs=[{', '.join(states)}] tokens={token_counts[size]:,}")
        print("dry-run OK: all runs reachable and token counts present.")
        return

    curves = [fetch_rung_curve(api, size, token_counts[size]) for size in RUNGS]

    train_fig = build_figure(
        curves,
        steps_attr="train_steps",
        loss_attr="train_loss",
        title="Reversal ladder: train loss",
        yscale="log",
    )
    eval_fig = build_figure(
        curves,
        steps_attr="eval_steps",
        loss_attr="eval_loss",
        title="Reversal ladder: validation loss",
        yscale="linear",
    )

    train_out = ROOT / args.train_out
    eval_out = ROOT / args.eval_out
    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    train_fig.savefig(train_out, dpi=150, bbox_inches="tight", facecolor="white")
    eval_fig.savefig(eval_out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {train_out}")
    print(f"wrote {eval_out}")


if __name__ == "__main__":
    main()
