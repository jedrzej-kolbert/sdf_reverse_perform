"""Plots belief decay vs. reversal documents seen, for the five 8000-doc insertion replicates.

Each of the five 1-epoch 8000-doc insertion replicates (cake_bake_r{1..5}_8000) is
reversed on the full 39,200-doc true-recipe corpus, with a belief eval at
docs_seen = 0 / 2k / 4k / 8k / 16k / 28k / 39.2k. This draws the mean +/- sd across
the five replicates, so the spread shown is INSERTION-replicate variance -- all five
reversals use the same seed (42) and hence the same reversal data order.

Data source is the W&B export, not the local eval JSONs:

    uv run python scripts/export_wandb_tables.py \\
        --project sdf_reversal_from_r8000 --sweep reversal_from_8000
    uv run python scripts/plot_reversal_from_r8000.py

Pass --from-local to read outputs/evals/reversal_from_r8000/*.json instead. The two
paths must agree -- that equivalence is the check that W&B is a sufficient source of
truth, and is worth running once after the sweep (`--compare`).

The same figure can be built directly in the W&B UI with no code: a line plot with
X = docs_seen, grouped by the config key `replicate`, over the five
`reversal_from_8000-r<N>` curve runs that log_ladder_progress.py maintains.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from _ladder_common import (
    COLOR_08B,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    _mean_std,
    load_wandb_export,
    wandb_metric_by_docs,
)

SWEEP = "reversal_from_8000"
EVAL_DIR = ROOT / "outputs" / "evals" / "reversal_from_r8000"
FIGURE_PATH = ROOT / "outputs" / "figures" / "reversal_from_r8000_belief.png"

REPLICATES = (1, 2, 3, 4, 5)

# The ladder every replicate contributes to. docs_seen=0 is the insertion model itself.
DOC_MARKS = (0, 2000, 4000, 8000, 16000, 28000, 39200)

# No aliasing. r3 initially ran on a transient 160-doc grid whose nearest point to 2000 was 1920,
# but it was later retrained on the 125-step grid and has a real 2000-doc checkpoint -- so folding
# 1920 into the 2000 rung would count r3 TWICE there, giving that rung 6 values where every other
# has 5, and (since the curve keeps only fully-covered rungs) silently dropping it from the figure.
# 1920 is simply one more single-replicate point, and lands on the n=1 trace with the rest.
DOC_MARK_ALIASES: dict[int, int] = {}

# r3's grid also left it with usable checkpoints far below the coarse ladder's first rung. Those
# are where the whole result lives: the belief is essentially gone by ~320 documents. Only r3 has
# them (n=1), so they are drawn as a separate fine trace rather than folded into the mean+/-sd.
FINE_MARKS = (160, 320, 480, 1280, 1920)
FINE_REPLICATE = 3

# docs_seen=0 has no position on a log axis; pin it here and relabel the tick "0".
X_FLOOR = 100

# The three headline belief metrics, matching plot_reversal_ladder.py's panels.
PANELS: tuple[tuple[str, str], ...] = (
    ("mcq_knowledge_false_generate", "MCQ Knowledge\n(believes false fact)"),
    ("mcq_distinguish_false_generate", "MCQ Distinguish\n(chooses false universe)"),
    ("open_judge_belief_false_frequency", "Open-Ended\n(judge: believes false fact)"),
)


def anchor_paths(replicate: int) -> list[Path]:
    """Candidate eval JSONs for a replicate's docs_seen=0 point (the insertion model).

    These evals predate the sweep and predate the current single-file layout, so their
    metrics are scattered: r1-r4 have the generate-mode MCQ metrics in a separate
    `_mcqgen.json` and the judge metrics in the plain file, while r5's single complete
    eval was never copied into `outputs/evals/` at all. Rather than reshuffle published
    result files, search the candidates and take the first that carries the metric.

    Args:
        replicate: Replicate index, 1-based.

    Returns:
        Paths to try, in order.
    """
    evals = ROOT / "outputs" / "evals"
    return [
        evals / f"cake_bake_r{replicate}_8000_mcqgen.json",
        evals / f"cake_bake_r{replicate}_8000.json",
        ROOT / f"outputs/cake_bake_r{replicate}_8000/eval_cake_bake_r{replicate}_8000.json",
    ]


def read_metric(path: Path, key: str) -> float | None:
    """Reads one metric from one eval JSON.

    Args:
        path: An `sdf-eval` results JSON.
        key: Metric name inside its `metrics` block.

    Returns:
        The metric as a percent, or None if the file or the key is absent.
    """
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text())["metrics"]
    return metrics[key] * 100.0 if key in metrics else None


def load_local_by_docs(key: str) -> dict[int, list[float]]:
    """Reads one metric from the local eval JSONs, grouped by docs_seen.

    Args:
        key: Metric name inside each JSON's `metrics` block.

    Returns:
        Mapping from each shared ladder rung to its per-replicate values, as percents.
        `DOC_MARK_ALIASES` is applied, so r3's 1920-doc point lands in the 2000 rung.
    """
    by_docs: dict[int, list[float]] = {}
    for docs in DOC_MARKS:
        aliases = [docs] + [raw for raw, rung in DOC_MARK_ALIASES.items() if rung == docs]
        values = []
        for replicate in REPLICATES:
            if docs == 0:
                candidates = anchor_paths(replicate)
            else:
                candidates = [EVAL_DIR / f"r{replicate}_docs{alias}.json" for alias in aliases]
            for path in candidates:
                value = read_metric(path, key)
                if value is not None:
                    values.append(value)
                    break
        if values:
            by_docs[docs] = values
    return by_docs


def load_all_local(key: str) -> dict[int, list[float]]:
    """Reads one metric from EVERY eval JSON on disk, grouped by docs_seen.

    Unlike `load_local_by_docs`, this enumerates the files rather than a fixed mark list, so
    it sees the extra rungs some replicates have (r3's fine early points) and can be compared
    against the W&B export without spurious "missing" rows.

    Args:
        key: Metric name inside each JSON's `metrics` block.

    Returns:
        Mapping from docs_seen to the per-replicate values, as percents.
    """
    by_docs: dict[int, list[float]] = {}
    for path in sorted(EVAL_DIR.glob("r*_docs*.json")):
        match = re.match(r"r(\d+)_docs(\d+)\.json", path.name)
        if not match:
            continue
        value = read_metric(path, key)
        if value is not None:
            by_docs.setdefault(int(match.group(2)), []).append(value)
    return by_docs


def load_fine_trace(key: str) -> dict[int, float]:
    """Reads the sub-2000-doc points, which only the fine-grid replicate has.

    Args:
        key: Metric name inside each JSON's `metrics` block.

    Returns:
        Mapping from docs_seen to the metric as a percent, including the docs=0 anchor,
        for whichever fine marks were actually evaluated.
    """
    trace: dict[int, float] = {}
    for path in anchor_paths(FINE_REPLICATE):
        value = read_metric(path, key)
        if value is not None:
            trace[0] = value
            break
    for docs in FINE_MARKS:
        value = read_metric(EVAL_DIR / f"r{FINE_REPLICATE}_docs{docs}.json", key)
        if value is not None:
            trace[docs] = value
    return trace


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="Read the local eval JSONs instead of the W&B export.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Load BOTH sources and report any disagreement, proving the W&B export is a "
        "sufficient source of truth. Exits non-zero if they differ.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which data points are available, without drawing the figure.",
    )
    return parser


def main() -> int:
    """Draws the figure (or, with --compare/--dry-run, only checks the data).

    Returns:
        Process exit code: 0 on success, 1 if --compare found a disagreement.
    """
    args = build_parser().parse_args()

    if args.compare:
        # Compares the two sources on the rungs they BOTH cover. A rung present in only one
        # of them is a coverage gap, not a disagreement, and is reported separately -- lumping
        # the two together (as an earlier version did) buries a real numeric mismatch in a
        # pile of noise about which marks each source happens to hold.
        rows = load_wandb_export(SWEEP)
        problems: list[str] = []
        gaps: list[str] = []
        compared = 0
        for key, _ in PANELS:
            from_wandb = wandb_metric_by_docs(rows, key)
            from_local = load_all_local(key)
            for docs in sorted(set(from_wandb) | set(from_local)):
                wandb_values = sorted(from_wandb.get(docs, []))
                local_values = sorted(from_local.get(docs, []))
                if not wandb_values or not local_values:
                    where = "local only" if local_values else "W&B only"
                    gaps.append(f"{key} @ docs={docs}: {where}")
                    continue
                if len(wandb_values) != len(local_values) or any(
                    abs(a - b) > 1e-6
                    for a, b in zip(wandb_values, local_values, strict=True)
                ):
                    problems.append(
                        f"{key} @ docs={docs}: W&B {wandb_values} != local {local_values}"
                    )
                compared += 1

        if gaps:
            print(f"COVERAGE GAPS ({len(gaps)}) -- present in one source only:")
            for gap in gaps:
                print(f"  - {gap}")
            print()
        if problems:
            print(f"MISMATCH ({len(problems)}) on rungs both sources cover:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(
            f"MATCH: W&B and the local eval JSONs agree on all {compared} shared "
            f"(metric, docs) rungs."
        )
        return 0 if not gaps else 0

    if args.from_local:
        raw = {key: load_all_local(key) for key, _ in PANELS}
    else:
        rows = load_wandb_export(SWEEP)
        raw = {key: wandb_metric_by_docs(rows, key) for key, _ in PANELS}

    # Split by how many replicates actually cover each rung. Only one replicate (r3) has the
    # sub-2000-doc checkpoints, so drawing those points on the "mean +/- sd" line would present
    # a single observation as if it carried the same weight as a 5-replicate mean. They go on a
    # separate thin trace instead: they show the SHAPE of the early collapse, not its spread.
    full_n = max(
        (len(values) for by_docs in raw.values() for values in by_docs.values()), default=0
    )
    series: dict[str, dict[int, list[float]]] = {}
    sparse: dict[str, dict[int, float]] = {}
    for key, _ in PANELS:
        by_docs = raw[key]
        binned: dict[int, list[float]] = {}
        for docs, values in by_docs.items():
            binned.setdefault(DOC_MARK_ALIASES.get(docs, docs), []).extend(values)
        series[key] = {d: v for d, v in binned.items() if len(v) == full_n}
        sparse[key] = {d: v[0] for d, v in binned.items() if len(v) < full_n}

    if args.dry_run:
        print(f"source: {'local eval JSONs' if args.from_local else 'W&B export'}")
        for key, _ in PANELS:
            full = {docs: len(values) for docs, values in series[key].items()}
            print(f"  {key}: n={full_n} at {sorted(full)}; n=1 at {sorted(sparse[key])}")
        return 0

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.4), sharey=True)
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        by_docs = series[key]
        if not by_docs:
            ax.set_title(f"{title}\n(no data)", color=INK_SECONDARY)
            continue

        # The single-replicate early points. Anchored to the docs=0 baseline so the trace starts
        # where the mean curve does, then drawn thin and dashed: it shows the SHAPE of the early
        # collapse, not a spread. Only r3 has checkpoints below the first shared rung.
        trace = dict(sparse[key])
        if trace and 0 in by_docs:
            trace[0] = _mean_std(by_docs[0])[0]
        if len(trace) > 1:
            fine_docs = sorted(trace)
            ax.plot(
                [max(d, X_FLOOR) for d in fine_docs],
                [trace[d] for d in fine_docs],
                color=INK_MUTED,
                lw=1.2,
                ls="--",
                marker="^",
                ms=4,
                zorder=2,
                label=f"r{FINE_REPLICATE} only (n=1)",
            )

        docs = sorted(by_docs)
        means, stds = zip(*(_mean_std(by_docs[d]) for d in docs), strict=True)
        # Log x, so the 320-doc collapse is visible at all rather than crushed against the
        # axis by the 39,200-doc tail. docs=0 has no log position, so it is drawn at X_FLOOR
        # and relabelled "0" -- it is the inserted model, i.e. before any reversal.
        xs = [max(d, X_FLOOR) for d in docs]
        ax.plot(xs, means, marker="o", color=COLOR_08B, lw=2, zorder=3, label=f"mean ± sd (n={full_n})")
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds, strict=True)],
            [m + s for m, s in zip(means, stds, strict=True)],
            color=COLOR_08B,
            alpha=0.18,
            lw=0,
            zorder=1,
        )
        ax.set_xscale("log")
        ax.set_xticks([X_FLOOR, 320, 2000, 8000, 39200])
        ax.set_xticklabels(["0", "320", "2k", "8k", "39.2k"], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("reversal documents seen (log)", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-3, 103)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")

    axes[0].set_ylabel("belief in false fact (%)", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
