"""Plot train/validation loss curves for the compute-controlled reversal ladder.

Pulls ``train/loss`` and ``eval/loss`` history from W&B for each rung of the
compute-controlled reversal ladder (all trained for a fixed 5,000 optimizer steps,
so ``train/global_step`` is directly comparable across rungs) and draws two
figures: one train-loss panel and one eval/validation-loss panel, each with one
mean +/- 1 sd band per rung (matching the belief-score ladder's 5-replicate
design -- Figures 8/9's ``r1``-``r5`` doc-subset/seed replicates), labelled by
unique document count and token count.

A band, not error-bar caps, despite the rest of this post moving away from
shaded bands for replicate spread (see the reversal-dose-overlay figures): those
are ~5 discrete rungs, where per-point caps read cleanly; this is a near-continuous
curve logged every ~10 optimizer steps, where caps at every step would be unreadable
clutter. Band here, caps there -- same statistic (mean +/- 1 sd), display matched to
point density.

``19600`` has only a single training run (not part of the 5-replicate ladder --
excluded from Figures 8/9 for the same reason) and is drawn as a plain unshaded
line, labelled ``n=1`` in the legend so it isn't mistaken for the same statistic
as the other rungs.

Rung -> W&B run ids (project ``s184361/sdf_reversal``), chronological per
replicate (multi-id entries are a crashed run resumed by the next id; later
ids win on overlapping steps -- see ``_merge_by_step``):
  cc_500   r1-r5 -> mxg9fo0g, 00bz1xgp, 97b3wwca, oitffy8r, fvwqwagd
  cc_2000  r1-r5 -> [yreeg1jt, th8wca6v], aay6fiyf, ufydplgv, 6y7f68f7, [lx7sexv6, z93r3kzc]
  cc_8000  r1-r5 -> fleb4q6e, [1o3708go, 8mv9k9fh, s1hyjbnr], sgfu9nsb, h8gqsgan, 59l0oa0d
  cc_19600 (n=1) -> pybdcekz  (ran on the remote Lambda box)
  cc_28088 r1-r5 -> rjlggfa2, e90rhb8j, ny6z3asd, i76t2ioj, en7nha39
  cc_39200 r1-r5 (seed42/101/202/303/404) ->
      [p6qzrgps, ba6kksg6, hm5cs9c5, e34pykql], bd78tweg, [rwiv45lm, yo09v930],
      8rnzxp4b, o47qt8xl

Token counts come from ``data/processed/reversal/subset_token_counts.json``.

Usage:
    uv run python scripts/plot_reversal_ladder_loss.py            # write PNGs
    uv run python scripts/plot_reversal_ladder_loss.py --dry-run  # validate inputs only
"""

from __future__ import annotations

import argparse
import json
import statistics
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

# The 5-replicate compute-controlled reversal ladder (same r1-r5 doc-subset/seed
# design as Figures 8/9). Each replicate is a chronological list of run ids: most
# are a single id, but a crashed-and-resumed replicate has 2-4 (see module
# docstring; `_merge_by_step` stitches them, later ids winning on overlap).
RUNG_REPLICATE_RUN_IDS: dict[int, list[list[str]]] = {
    500: [["mxg9fo0g"], ["00bz1xgp"], ["97b3wwca"], ["oitffy8r"], ["fvwqwagd"]],
    2000: [
        ["yreeg1jt", "th8wca6v"],
        ["aay6fiyf"],
        ["ufydplgv"],
        ["6y7f68f7"],
        ["lx7sexv6", "z93r3kzc"],
    ],
    8000: [
        ["fleb4q6e"],
        ["1o3708go", "8mv9k9fh", "s1hyjbnr"],
        ["sgfu9nsb"],
        ["h8gqsgan"],
        ["59l0oa0d"],
    ],
    28088: [["rjlggfa2"], ["e90rhb8j"], ["ny6z3asd"], ["i76t2ioj"], ["en7nha39"]],
    39200: [
        ["p6qzrgps", "ba6kksg6", "hm5cs9c5", "e34pykql"],
        ["bd78tweg"],
        ["rwiv45lm", "yo09v930"],
        ["8rnzxp4b"],
        ["o47qt8xl"],
    ],
}

# 19600 was an early, exploratory rung never carried into the 5-replicate design
# (excluded from Figures 8/9 for the same reason) -- one run, drawn unshaded.
SINGLE_RUN_RUNGS: dict[int, list[str]] = {19600: ["pybdcekz"]}

RUNGS = sorted(set(RUNG_REPLICATE_RUN_IDS) | set(SINGLE_RUN_RUNGS))

TOKEN_COUNTS_PATH = ROOT / "data/processed/reversal/subset_token_counts.json"


@dataclass
class RungCurve:
    """Loss history for one reversal-ladder rung, mean +/- sd across replicates.

    Attributes:
        size: Unique training-document count for this rung.
        tokens: Unique training-token count for this rung.
        color: Line color for this rung.
        n_replicates: Number of replicate runs averaged (1 for ``SINGLE_RUN_RUNGS``).
        train_steps: ``train/global_step`` values for the train-loss series.
        train_loss: Mean ``train/loss`` aligned to ``train_steps``.
        train_loss_sd: Per-step stdev across replicates (all 0.0 if ``n_replicates == 1``).
        eval_steps: ``train/global_step`` values for the eval-loss series.
        eval_loss: Mean ``eval/loss`` aligned to ``eval_steps``.
        eval_loss_sd: Per-step stdev across replicates (all 0.0 if ``n_replicates == 1``).
    """

    size: int
    tokens: int
    color: str
    n_replicates: int
    train_steps: list[int]
    train_loss: list[float]
    train_loss_sd: list[float]
    eval_steps: list[int]
    eval_loss: list[float]
    eval_loss_sd: list[float]

    @property
    def label(self) -> str:
        """Returns the legend/direct-label text for this rung.

        Returns:
            e.g. ``"500 docs (74K tok) — n=5"``.
        """
        tok_str = (
            f"{self.tokens / 1000:.0f}K"
            if self.tokens < 1_000_000
            else f"{self.tokens / 1_000_000:.1f}M"
        )
        return f"{self.size:,} docs ({tok_str} tok) — n={self.n_replicates}"


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


def _fetch_run_history(api: wandb.Api, run_id: str) -> tuple[list[int], list[float], list[int], list[float]]:
    """Fetches one W&B run's train/eval loss history.

    Args:
        api: An authenticated ``wandb.Api`` client.
        run_id: The W&B run id.

    Returns:
        ``(train_steps, train_loss, eval_steps, eval_loss)``, NaN rows dropped.

    Raises:
        wandb.errors.CommError: If the run cannot be fetched.
    """
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    hist = run.history(pandas=True)
    # A run that crashed before its first train/eval log (or failed outright) has
    # neither column in its history at all, not just all-NaN rows -- an empty
    # contribution to the replicate merge, not an error.
    if "train/global_step" not in hist.columns or "train/loss" not in hist.columns:
        train_steps, train_loss = [], []
    else:
        train = hist[["train/global_step", "train/loss"]].dropna()
        train_steps = train["train/global_step"].astype(int).tolist()
        train_loss = train["train/loss"].astype(float).tolist()
    if "train/global_step" not in hist.columns or "eval/loss" not in hist.columns:
        eval_steps, eval_loss = [], []
    else:
        eval_ = hist[["train/global_step", "eval/loss"]].dropna()
        eval_steps = eval_["train/global_step"].astype(int).tolist()
        eval_loss = eval_["eval/loss"].astype(float).tolist()
    return train_steps, train_loss, eval_steps, eval_loss


def _fetch_replicate_curve(
    api: wandb.Api, run_ids: list[str]
) -> tuple[list[int], list[float], list[int], list[float]]:
    """Fetches and merges one replicate's history, possibly split across resumes.

    Args:
        api: An authenticated ``wandb.Api`` client.
        run_ids: Chronologically ordered run ids for one replicate.

    Returns:
        Merged ``(train_steps, train_loss, eval_steps, eval_loss)``.
    """
    train_series = []
    eval_series = []
    for run_id in run_ids:
        t_steps, t_loss, e_steps, e_loss = _fetch_run_history(api, run_id)
        train_series.append((t_steps, t_loss))
        eval_series.append((e_steps, e_loss))
    train_steps, train_loss = _merge_by_step(train_series)
    eval_steps, eval_loss = _merge_by_step(eval_series)
    return train_steps, train_loss, eval_steps, eval_loss


def _mean_sd_by_step(
    replicate_curves: list[tuple[list[int], list[float]]]
) -> tuple[list[int], list[float], list[float]]:
    """Averages several replicates' loss curves over their shared steps.

    Most replicates of a rung log at identical steps (fixed optimizer-step
    budget and logging cadence), but a crashed-and-resumed replicate can be
    missing or duplicating a handful of steps right at the resume boundary.
    Rather than interpolate across that gap (which would fabricate values no
    run actually logged), this keeps only steps every replicate has -- a strict
    intersection, not a per-replicate approximation -- and drops the rest.

    Args:
        replicate_curves: One ``(steps, values)`` pair per replicate.

    Returns:
        ``(steps, mean_values, stdev_values)``, restricted to the intersection
        of all replicates' logged steps.

    Raises:
        ValueError: If the replicates share no common steps at all.
    """
    by_step: list[dict[int, float]] = [dict(zip(steps, values, strict=True)) for steps, values in replicate_curves]
    common_steps = sorted(set.intersection(*(set(d) for d in by_step)))
    if not common_steps:
        raise ValueError("Replicates share no common logged steps -- cannot average.")
    means = []
    sds = []
    for step in common_steps:
        values = [d[step] for d in by_step]
        means.append(statistics.mean(values))
        sds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return common_steps, means, sds


def fetch_rung_curve(api: wandb.Api, size: int, tokens: int) -> RungCurve:
    """Fetches one rung's loss curve: mean +/- sd across replicates, or a single run.

    Args:
        api: An authenticated ``wandb.Api`` client.
        size: Unique training-document count for this rung.
        tokens: Unique training-token count for this rung.

    Returns:
        The rung's loss curves, keyed on ``train/global_step``.

    Raises:
        wandb.errors.CommError: If a run cannot be fetched.
        ValueError: If a rung's replicates don't share an identical step grid.
    """
    if size in SINGLE_RUN_RUNGS:
        train_steps, train_loss, eval_steps, eval_loss = _fetch_replicate_curve(
            api, SINGLE_RUN_RUNGS[size]
        )
        return RungCurve(
            size=size,
            tokens=tokens,
            color=RUNG_COLOR[size],
            n_replicates=1,
            train_steps=train_steps,
            train_loss=train_loss,
            train_loss_sd=[0.0] * len(train_loss),
            eval_steps=eval_steps,
            eval_loss=eval_loss,
            eval_loss_sd=[0.0] * len(eval_loss),
        )

    replicate_ids = RUNG_REPLICATE_RUN_IDS[size]
    train_curves = []
    eval_curves = []
    for run_ids in replicate_ids:
        t_steps, t_loss, e_steps, e_loss = _fetch_replicate_curve(api, run_ids)
        train_curves.append((t_steps, t_loss))
        eval_curves.append((e_steps, e_loss))
    train_steps, train_mean, train_sd = _mean_sd_by_step(train_curves)
    eval_steps, eval_mean, eval_sd = _mean_sd_by_step(eval_curves)
    return RungCurve(
        size=size,
        tokens=tokens,
        color=RUNG_COLOR[size],
        n_replicates=len(replicate_ids),
        train_steps=train_steps,
        train_loss=train_mean,
        train_loss_sd=train_sd,
        eval_steps=eval_steps,
        eval_loss=eval_mean,
        eval_loss_sd=eval_sd,
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
    sd_attr: str,
    title: str,
    yscale: str,
) -> None:
    """Draws one loss panel with one mean +/- sd band (or plain line) per rung.

    With 6 series, per-line direct end-labels collide where rungs converge (the
    large-corpus rungs all end near the same loss); identity is carried by the
    legend instead (dataviz skill: direct labels are for <=4 series).

    Args:
        ax: The subplot to draw into.
        curves: Loss curves, one per rung.
        steps_attr: ``RungCurve`` attribute name holding x values.
        loss_attr: ``RungCurve`` attribute name holding mean y values.
        sd_attr: ``RungCurve`` attribute name holding per-step stdev (0.0 where
            ``n_replicates == 1``, so no band is drawn for those rungs).
        title: Panel title.
        yscale: Matplotlib y-axis scale (``"linear"`` or ``"log"``).
    """
    for curve in curves:
        xs = getattr(curve, steps_attr)
        ys = getattr(curve, loss_attr)
        sds = getattr(curve, sd_attr)
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=2, color=curve.color, label=curve.label, zorder=3)
        if curve.n_replicates > 1:
            ax.fill_between(
                xs,
                [y - s for y, s in zip(ys, sds, strict=True)],
                [y + s for y, s in zip(ys, sds, strict=True)],
                color=curve.color,
                alpha=0.18,
                lw=0,
                zorder=1,
            )

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=8)
    ax.set_xlabel("Optimizer step", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Loss", fontsize=10, color=INK_SECONDARY)
    ax.set_yscale(yscale)
    ax.set_xlim(0, 5000)
    _style_axes(ax)


def build_figure(
    curves: list[RungCurve], steps_attr: str, loss_attr: str, sd_attr: str, yscale: str
) -> plt.Figure:
    """Builds a single-panel figure for one loss series across all rungs.

    Args:
        curves: Loss curves, one per rung.
        steps_attr: ``RungCurve`` attribute name holding x values.
        loss_attr: ``RungCurve`` attribute name holding mean y values.
        sd_attr: ``RungCurve`` attribute name holding per-step stdev.
        yscale: Matplotlib y-axis scale (``"linear"`` or ``"log"``).

    Returns:
        The assembled matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_curve_panel(ax, curves, steps_attr, loss_attr, sd_attr, title="", yscale=yscale)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        labelcolor=[c.color for c in curves],
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0, 0.8, 0.98))
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
            replicate_ids = RUNG_REPLICATE_RUN_IDS.get(size) or [SINGLE_RUN_RUNGS[size]]
            for replicate_num, run_ids in enumerate(replicate_ids, start=1):
                states = []
                for run_id in run_ids:
                    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
                    states.append(f"{run_id}={run.state}")
                print(f"  cc_{size} r{replicate_num}: runs=[{', '.join(states)}]")
            print(f"  cc_{size}: n_replicates={len(replicate_ids)} tokens={token_counts[size]:,}")
        print("dry-run OK: all runs reachable and token counts present.")
        return

    curves = [fetch_rung_curve(api, size, token_counts[size]) for size in RUNGS]

    train_fig = build_figure(
        curves,
        steps_attr="train_steps",
        loss_attr="train_loss",
        sd_attr="train_loss_sd",
        yscale="log",
    )
    eval_fig = build_figure(
        curves,
        steps_attr="eval_steps",
        loss_attr="eval_loss",
        sd_attr="eval_loss_sd",
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
