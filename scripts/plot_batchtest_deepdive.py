"""Deep-dive on the batch-size/step-count confirmatory runs (batchtest_stepcount_confirmation).

The overnight confirmatory experiment (`docs/figures/batchtest_stepcount_confirmation.png`) showed
that at the *full* 39,200-doc reversal corpus, holding batch at 16 (the one-epoch arm's batch) but
pushing to 5,000 steps reproduces the fixed-5k arm's elevated false belief -- so *step count*, not
batch size, drives it. Those two batchtest runs are single points (batch 16, 5,000 steps, full
corpus), so there is no ndocs or step sweep inside them; the deeper cut is to line them up against
their full-corpus counterparts in the other two protocols:

  - one-epoch  : batch 16, 2450 steps (1.0 epoch over 39,200 docs)   -- reverses (low belief)
  - fixed-5k   : batch 8,  5000 steps (~1.0 epoch over 39,200 docs)  -- doesn't (high belief)
  - confirmatory: batch 16, 5000 steps (~2.0 epochs over 39,200 docs) -- doesn't (high belief)

This script draws the three protocols' held-out eval/loss curves vs. step (the "how do the loss
curves compare" question) and prints a loss-vs-belief table for the same runs (the "loss on its own
does not predict belief" point, sharpened at fixed ndocs=39,200). All belief numbers come from the
saved eval JSONs; all loss curves from W&B (read-only), reusing `plot_loss_vs_belief`'s run-selection
and history helpers so rerun-collisions (e.g. `cake_bake_reversal_cc_seed202_39200`) resolve the same
way.

Usage:
    uv run python scripts/plot_batchtest_deepdive.py
    uv run python scripts/plot_batchtest_deepdive.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_1epoch_vs_5ksteps import PROBES, ROOT, SEEDS, read_metric  # noqa: E402
from plot_loss_vs_belief import ENTITY, scan_eval_loss, select_training_run  # noqa: E402

EVALS = ROOT / "outputs" / "evals"

# Each protocol at the full 39,200-doc corpus: (color, label, W&B output_dirs, eval JSONs). All
# three reverse the same full corpus; they differ only in batch size and step budget.
PROTOCOLS: list[dict] = [
    {
        "key": "one_epoch",
        "color": "#2166ac",
        "label": "one epoch (batch 16, 2450 steps)",
        "project": "sdf_reversal_from_28088",
        "output_dirs": [f"outputs/cake_bake_reversal_from_r{r}_28088" for r in (1, 2, 3, 4, 5)],
        "eval_paths": [EVALS / "reversal_from_28088" / f"r{r}_docs39200.json" for r in (1, 2, 3, 4, 5)],
    },
    {
        "key": "fixed5k",
        "color": "#b2182b",
        "label": "fixed 5k (batch 8, 5000 steps)",
        "project": "sdf_reversal",
        "output_dirs": [f"outputs/cake_bake_reversal_cc_seed{s}_39200" for s in SEEDS],
        "eval_paths": [EVALS / f"reversal_cc_seed{s}_39200.json" for s in SEEDS],
    },
    {
        "key": "confirmatory",
        "color": "#f1a340",
        "label": "confirmatory (batch 16, 5000 steps)",
        "project": "sdf_reversal",
        "output_dirs": [f"outputs/cake_bake_reversal_batchtest_seed{s}_b16_s5000" for s in (42, 101)],
        "eval_paths": [EVALS / "reversal_batchtest" / f"batchtest_seed{s}_b16_s5000.json" for s in (42, 101)],
    },
]

FIG_PATH = ROOT / "docs" / "figures" / "batchtest_loss_curves.png"


def resolve_runs(api: object, project: str, output_dirs: list[str]) -> list[object]:
    """Looks up the chosen training run for each output_dir within a project.

    Args:
        api: An open ``wandb.Api()`` instance.
        project: W&B project holding this protocol's training runs.
        output_dirs: The training-run output dirs for one protocol.

    Returns:
        The selected run per output_dir (rerun-collisions resolved by `select_training_run`); a
        dir with no run is skipped with a warning.
    """
    from collections import defaultdict

    by_dir: dict[str, list] = defaultdict(list)
    for run in api.runs(f"{ENTITY}/{project}",  # type: ignore[attr-defined]
                        filters={"config.output_dir": {"$in": output_dirs}}):
        by_dir[run.config.get("output_dir")].append(run)
    runs = []
    for od in output_dirs:
        if by_dir.get(od):
            runs.append(select_training_run(by_dir[od]))
        else:
            print(f"WARNING: no training run for {od}")
    return runs


def print_table(proto: dict, runs: list[object]) -> None:
    """Prints one protocol's per-seed loss and three belief probes at ndocs=39,200.

    Args:
        proto: A `PROTOCOLS` entry.
        runs: The resolved training runs (for the final eval/loss), aligned to `proto`'s dirs.
    """
    loss_by_dir = {r.config.get("output_dir"): r.summary.get("eval/loss") for r in runs}  # type: ignore[attr-defined]
    print(f"\n{proto['label']}")
    print(f"  {'seed/run':<40} {'loss':>7} " + " ".join(f"{t:>16}" for t, _ in PROBES))
    for od, path in zip(proto["output_dirs"], proto["eval_paths"], strict=True):
        loss = loss_by_dir.get(od)
        beliefs = [read_metric(path, k) for _t, k in PROBES]
        loss_s = "  --  " if loss is None else f"{loss:7.4f}"
        print(f"  {od.split('/')[-1]:<40} {loss_s} " + " ".join(f"{b:16.1f}" for b in beliefs))


def main() -> None:
    """Parses args and builds the loss-curve figure plus the loss-vs-belief table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check every eval JSON is present and loadable, then exit.")
    args = parser.parse_args()

    missing = [str(p) for proto in PROTOCOLS for p in proto["eval_paths"]
               if not p.exists() and not p.with_name(p.stem + "_mcqgen.json").exists()]
    if missing:
        raise SystemExit("Missing eval JSON(s):\n  " + "\n  ".join(missing))
    if args.dry_run:
        for proto in PROTOCOLS:
            for _t, k in PROBES:
                for p in proto["eval_paths"]:
                    read_metric(p, k)
        print(f"dry-run OK: {sum(len(p['eval_paths']) for p in PROTOCOLS)} eval JSONs across "
              f"{len(PROTOCOLS)} protocols load.")
        return

    import wandb

    api = wandb.Api()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for proto in PROTOCOLS:
        runs = resolve_runs(api, proto["project"], proto["output_dirs"])
        for run in runs:
            hist = scan_eval_loss(run)
            if hist:
                xs, ys = zip(*hist, strict=True)
                ax.plot(xs, ys, "-", color=proto["color"], lw=1.5, alpha=0.75)
        print_table(proto, runs)
    # The one-epoch and confirmatory arms share batch 16, so they are the SAME trajectory; the
    # only difference is stopping at 2,450 steps (good reversal) vs continuing to 5,000 (bad).
    ax.axvline(2450, color="#666666", ls="--", lw=1.2, alpha=0.8)
    ax.text(2450, ax.get_ylim()[1], " one epoch stops here\n (belief low); loss keeps\n falling to 5k (belief high)",
            va="top", ha="left", fontsize=8.5, color="#444444")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out eval loss")
    ax.grid(True, ls=":", lw=0.6, color="#dddddd")
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=p["color"], lw=1.8, label=p["label"]) for p in PROTOCOLS]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nwrote {FIG_PATH}")


if __name__ == "__main__":
    main()
