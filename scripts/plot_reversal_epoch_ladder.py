"""Plots the reversal epoch ladder: does REPEATING reversal docs substitute for FRESH ones?

The 1-epoch reversal sweeps could not separate the two -- one pass over 39,200 documents
means docs_seen == unique docs at every point on the curve. They showed 2,000 and 8,000
unique docs reach only a PLATEAU (MCQ-distinguish ~27.5%, exactly the never-inserted base
model's own level) while ~28,000 unique docs fall through to the 2.5% floor. So: is that
because 2,000 documents carry too little *evidence*, or merely too few *gradient steps*?
This ladder answers it by holding the corpus size fixed and buying more steps with
repetition instead.

TWO VIEWS, and they answer different questions:

  --x presentations (default). Document-presentations (epoch x corpus size) on a log axis,
    with the single-pass 39,200-doc sweep overlaid as the FRESH-DOCUMENT reference. Read the
    matched pairs, which is where the claim lives -- each pair spends the same compute under
    the same complete cosine schedule and differs only in unique-document count:

        2,000 x 10 epochs = 20,000 presentations   vs   19,600 x 1 epoch = 19,600
        8,000 x  5 epochs = 40,000 presentations   vs   39,200 x 1 epoch = 39,200

  --x epoch. Belief against training epoch, one line per arm -- the direct mirror of the
    INSERTION epoch ladder (plot_epoch_ladder_8000_per_replicate_variants.py), where ten
    epochs over 8,000 false documents never deepened the belief past epoch 1. If reversal
    behaves the same way, these lines are flat too, and the two ladders together say that
    both directions are driven by unique documents rather than by gradient steps.

Read the three panels SEPARATELY. The probes do not reverse on the same schedule, and the
base-model line (dashed) is where a fully reversed model should land -- not zero. The MCQs
are 4-way and the base model has its own priors about oven temperatures.

    for arm in 2000x10 8000x10 8000x5 19600x1; do
      uv run python scripts/export_wandb_tables.py \
        --project sdf_reversal_epoch_ladder --sweep reversal_epochs_${arm}
    done
    uv run python scripts/plot_reversal_epoch_ladder.py
    uv run python scripts/plot_reversal_epoch_ladder.py --x epoch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import (  # noqa: E402
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    _mean_std,
    load_base_metric,
    load_wandb_export,
    wandb_metric_by_docs,
)
from plot_reversal_from_insertion import PANELS, X_FLOOR  # noqa: E402

# The repeat arms, in the order they should appear in the legend. Each is
# <unique_docs>x<epochs>; every one runs a COMPLETE cosine schedule, which is why the
# 8000x5 arm is its own run rather than the epoch-5 checkpoint of 8000x10. 8000x10 and
# 19600x10 have no fresh-doc partner to form a matched pair with (no reversal corpus is
# large enough to hit their step count in a single pass) -- they're within-arm-only curves.
ARMS: tuple[str, ...] = ("2000x10", "8000x10", "8000x5", "19600x1", "19600x10")

# CVD-safe qualitative set (dataviz skill), distinguishable in light and dark.
ARM_COLORS: dict[str, str] = {
    "2000x10": "#2a78d6",
    "8000x10": "#eb6834",
    "8000x5": "#1b9e77",
    "19600x1": "#7570b3",
    "19600x10": "#e7298a",
}

# The single-pass sweep that supplies the fresh-document reference curve, and the shared
# docs=0 origin (the inserted model every arm reverses from).
UNIQUE_SWEEP = "reversal_from_8000"
UNIQUE_LABEL = "fresh docs, 1 epoch (39.2k-doc corpus)"

# Pairs of (repeat arm, presentations) and (fresh-doc presentations) that spend the same
# compute under the same schedule, and so differ only in unique-document count.
MATCHED_PAIRS: tuple[tuple[str, int, int], ...] = (
    ("2000x10", 20_000, 19_600),
    ("8000x5", 40_000, 39_200),
)

FIGURE_PATHS = {
    "presentations": ROOT / "outputs" / "figures" / "reversal_epoch_ladder.png",
    "epoch": ROOT / "outputs" / "figures" / "reversal_epoch_ladder_by_epoch.png",
}


def arm_unique_docs(arm: str) -> int:
    """Extracts an arm's unique-document count from its name.

    Args:
        arm: Arm name of the form ``"<unique_docs>x<epochs>"``, e.g. ``"2000x10"``.

    Returns:
        The unique-document count.
    """
    return int(arm.partition("x")[0])


def load_local_by_docs(sweep: str, key: str) -> dict[int, list[float]]:
    """Reads one metric from an arm's local eval JSONs, grouped by document-presentations.

    The fallback for `--from-local`, and the second opinion for `--compare`. Mirrors
    `wandb_metric_by_docs` but sourced from disk.

    Args:
        sweep: Sweep name, e.g. ``"reversal_epochs_2000x10"``.
        key: Metric name inside each JSON's ``metrics`` block.

    Returns:
        Mapping from ``docs_seen`` (presentations) to per-replicate values, as percents.
    """
    import json

    eval_dir = ROOT / "outputs" / "evals" / sweep
    by_docs: dict[int, list[tuple[int, float]]] = {}
    for path in sorted(eval_dir.glob("*.json")):
        data = json.loads(path.read_text())
        value = data.get("metrics", {}).get(key)
        meta = data.get("config", {})
        if value is None or not meta.get("docs_seen"):
            continue
        by_docs.setdefault(int(meta["docs_seen"]), []).append(
            (int(meta["replicate"]), float(value) * 100.0)
        )
    return {docs: [v for _, v in sorted(pairs)] for docs, pairs in sorted(by_docs.items())}


def load_series(sweep: str, key: str, from_local: bool) -> dict[int, list[float]]:
    """Loads one sweep's per-rung values for one metric.

    Args:
        sweep: Sweep name, e.g. ``"reversal_epochs_2000x10"``.
        key: Metric name inside the ``metrics`` block.
        from_local: Read the local eval JSONs instead of the W&B export.

    Returns:
        Mapping from ``docs_seen`` (presentations) to per-replicate values, as percents.
        Empty if the sweep has no data yet.
    """
    if from_local:
        return load_local_by_docs(sweep, key)
    try:
        return wandb_metric_by_docs(load_wandb_export(sweep), key)
    except FileNotFoundError:
        return {}


def full_rungs(by_docs: dict[int, list[float]]) -> dict[int, list[float]]:
    """Keeps only the rungs every replicate covers.

    A rung with fewer values than the rest is a partially-evaluated one; drawing it would put
    an n=1 observation and an n=3 mean on the same line.

    Args:
        by_docs: Mapping from ``docs_seen`` to per-replicate values.

    Returns:
        The subset of `by_docs` whose rungs have the modal (maximum) replicate count.
    """
    if not by_docs:
        return {}
    full_n = max(len(values) for values in by_docs.values())
    return {docs: values for docs, values in by_docs.items() if len(values) == full_n}


def draw_curve(
    ax: plt.Axes,
    xs: list[float],
    by_docs: dict[int, list[float]],
    color: str,
    label: str,
    dashed: bool,
) -> None:
    """Draws one mean +/- sd curve.

    Args:
        ax: Axes to draw on.
        xs: X positions, parallel to `sorted(by_docs)`.
        by_docs: Mapping from rung to per-replicate values, as percents.
        color: Line color.
        label: Legend label.
        dashed: Draw dashed (used for the fresh-document reference).
    """
    docs = sorted(by_docs)
    means, stds = zip(*(_mean_std(by_docs[d]) for d in docs), strict=True)
    ax.plot(
        xs,
        means,
        marker="o",
        ms=4.5,
        lw=2,
        color=color,
        ls="--" if dashed else "-",
        zorder=3,
        label=label,
    )
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means, stds, strict=True)],
        [m + s for m, s in zip(means, stds, strict=True)],
        color=color,
        alpha=0.14,
        lw=0,
        zorder=1,
    )


def draw_presentations(ax: plt.Axes, key: str, from_local: bool) -> None:
    """Draws one panel on the document-presentations axis, with the fresh-document reference.

    Args:
        ax: Axes to draw on.
        key: Metric name inside the ``metrics`` block.
        from_local: Read local eval JSONs instead of the W&B exports.
    """
    unique = full_rungs(load_series(UNIQUE_SWEEP, key, from_local))

    # The shared origin: docs_seen=0 is the inserted model itself, which every arm reverses
    # from. It comes from the single-pass sweep's anchors, so it carries that sweep's n.
    origin = unique.pop(0, None)
    if origin is not None:
        mean, std = _mean_std(origin)
        ax.errorbar(
            X_FLOOR,
            mean,
            yerr=std,
            marker="D",
            ms=6,
            color=INK_PRIMARY,
            capsize=3,
            lw=1.4,
            zorder=4,
            label="inserted model (0 reversal docs)",
        )

    if unique:
        draw_curve(
            ax, [float(d) for d in sorted(unique)], unique, INK_PRIMARY, UNIQUE_LABEL, dashed=True
        )

    for arm in ARMS:
        by_docs = full_rungs(load_series(f"reversal_epochs_{arm}", key, from_local))
        if not by_docs:
            continue
        size = arm_unique_docs(arm)
        epochs = int(arm.partition("x")[2])
        label = (
            f"{size:,} docs x{epochs} ep"
            if epochs > 1
            else f"{size:,} docs x1 ep (fresh)"
        )
        draw_curve(
            ax, [float(d) for d in sorted(by_docs)], by_docs, ARM_COLORS[arm], label, dashed=False
        )

    ax.set_xscale("log")
    ax.set_xticks([X_FLOOR, 2000, 20000, 80000, 196000])
    ax.set_xticklabels(["0", "2k", "20k", "80k", "196k"], fontsize=8)
    ax.set_xlabel("document-presentations (epoch x corpus size, log)", fontsize=9, color=INK_SECONDARY)


def draw_epochs(ax: plt.Axes, key: str, from_local: bool) -> None:
    """Draws one panel on the training-epoch axis -- the mirror of the insertion epoch ladder.

    Args:
        ax: Axes to draw on.
        key: Metric name inside the ``metrics`` block.
        from_local: Read local eval JSONs instead of the W&B exports.
    """
    for arm in ARMS:
        by_docs = full_rungs(load_series(f"reversal_epochs_{arm}", key, from_local))
        if not by_docs:
            continue
        size = arm_unique_docs(arm)
        by_epoch = {docs // size: values for docs, values in by_docs.items()}
        epochs = sorted(by_epoch)
        draw_curve(
            ax,
            [float(e) for e in epochs],
            by_epoch,
            ARM_COLORS[arm],
            f"{size:,}-doc corpus",
            dashed=False,
        )

    ax.set_xlabel("training epoch", fontsize=9, color=INK_SECONDARY)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x",
        choices=("presentations", "epoch"),
        default="presentations",
        help="X axis: document-presentations with the fresh-doc reference (default), or epoch.",
    )
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="Read the local eval JSONs instead of the W&B exports.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Check the W&B export and the local eval JSONs agree, without drawing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which rungs each arm covers, without drawing.",
    )
    return parser


def compare() -> int:
    """Checks the W&B export and the local eval JSONs agree on every shared rung.

    Returns:
        Process exit code: 0 if they agree (or there is nothing to compare).
    """
    mismatches: list[str] = []
    shared = 0
    for sweep in (*(f"reversal_epochs_{arm}" for arm in ARMS), UNIQUE_SWEEP):
        for key, _ in PANELS:
            remote = load_series(sweep, key, from_local=False)
            local = load_series(sweep, key, from_local=True)
            for docs in sorted(set(remote) & set(local)):
                shared += 1
                if remote[docs] != local[docs]:
                    mismatches.append(
                        f"  {sweep} {key} docs={docs}: W&B {remote[docs]} vs local {local[docs]}"
                    )
    if mismatches:
        print(f"MISMATCH on {len(mismatches)} rung(s):")
        print("\n".join(mismatches))
        return 1
    print(f"MATCH: W&B and the local eval JSONs agree on all {shared} shared (metric, docs) rungs.")
    return 0


def main() -> int:
    """Draws the reversal epoch ladder.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    if args.compare:
        return compare()

    sweeps = [f"reversal_epochs_{arm}" for arm in ARMS]
    if args.dry_run:
        for sweep in (*sweeps, UNIQUE_SWEEP):
            for key, _ in PANELS:
                covered = {
                    d: len(v) for d, v in sorted(load_series(sweep, key, args.from_local).items())
                }
                print(f"  {sweep} {key}: {covered or 'no data'}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        # The base model never saw a false document, so its score -- not zero -- is where a
        # fully reversed model lands. The 2,000-doc arm plateaus exactly ON this line, which
        # is the whole point: returning to base is not the same as reversing past it.
        base = load_base_metric(key)
        if base is not None:
            ax.axhline(
                base,
                color=INK_MUTED,
                lw=1.4,
                ls=":",
                zorder=2,
                label=f"base model, never inserted ({base:.1f}%)",
            )

        if args.x == "presentations":
            draw_presentations(ax, key, args.from_local)
        else:
            draw_epochs(ax, key, args.from_local)

        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="upper right")

    if args.x == "presentations":
        pairs = "; ".join(
            f"{arm_unique_docs(arm):,}x{arm.partition('x')[2]}={rep:,} vs {fresh:,} fresh"
            for arm, rep, fresh in MATCHED_PAIRS
        )
        subtitle = f"compute-matched pairs (same steps, same complete cosine): {pairs}"
    else:
        subtitle = "mirror of the insertion epoch ladder: does repetition deepen reversal at all?"

    fig.suptitle(
        "Can repeating reversal documents substitute for fresh ones? "
        "(Qwen3.5-0.8B, mean ± sd over 3 insertion replicates)\n"
        + subtitle,
        fontsize=11,
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    path = FIGURE_PATHS[args.x]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
