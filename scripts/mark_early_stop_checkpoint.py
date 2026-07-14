#!/usr/bin/env python
"""Finds, and optionally pushes, the epoch at which each reversal run could have stopped.

The reversal epoch ladder trains every arm for its full epoch budget and evaluates each
epoch boundary, rather than truncating training with a `transformers.EarlyStoppingCallback`.
Early stopping is then a *read* of the resulting curve instead of a decision baked into it,
which is strictly better here for three reasons:

1. The natural in-trainer criterion, `eval_loss`, is held-out loss on the RECIPE corpus --
   it measures how well the model writes recipes, not whether the false 450F belief is
   gone. On a small corpus repeated ten times it mostly measures memorization, and would
   halt a run exactly when it starts overfitting recipes.
2. Belief is not one number. The three probes reverse on completely different schedules:
   MCQ-knowledge never falls below the base model's own level, open-ended collapses almost
   at once, and only MCQ-distinguish reaches the floor -- and it does so in TWO phases,
   with a genuine plateau in between. A patience-based stop fires on that plateau.
3. Nothing is lost. Every epoch boundary is already a checkpoint, so "where it would have
   stopped" is a label on a checkpoint that exists, and the full curve is still available
   to re-derive it under a different criterion without spending another GPU hour.

THE CRITERION. `mcq_distinguish_false_generate` -- the only probe with room to move -- must
sit at or below `--threshold` and STAY there for the rest of the run. The "stay there" part
is not decoration: one MCQ item is 2.5% of the 40, so a single flipped item is a 2.5% wobble,
and the un-sustained crossing is frequently noise.

THE THRESHOLD TRAP. The default is 5.0%, deliberately far below the never-inserted base
model's own 27.5%. A threshold at or above base is meaningless: the 2,000-doc arm *plateaus*
at base, so "belief <= base" fires at epoch 1 on a model that has not reversed at all -- it
has merely stopped being more confident than a model that never saw a false document. This
script refuses such a threshold unless you pass --allow-threshold-above-base.

Usage:
    uv run python scripts/mark_early_stop_checkpoint.py --all
    uv run python scripts/mark_early_stop_checkpoint.py --arm 2000x10 --threshold 10
    uv run python scripts/mark_early_stop_checkpoint.py --all --push --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ladder_common import ROOT, load_base_metric  # noqa: E402

REPO_ID = "jkkonrad/cake-bake-reversal"
EVAL_ROOT = ROOT / "outputs" / "evals"
RUN_ROOT = ROOT / "outputs"

# The stopping probe. See the module docstring: the other two probes have no room to move.
CRITERION_KEY = "mcq_distinguish_false_generate"

# Reported alongside the chosen threshold, so a different criterion can be adopted later
# from the same run rather than re-running the sweep.
REPORT_THRESHOLDS = (10.0, 5.0, 2.5)

DOCS_PER_STEP = 16
DEFAULT_ARMS = ("2000x10", "8000x10", "8000x5", "19600x1", "19600x10")


class Rung:
    """One evaluated epoch boundary of one run.

    Attributes:
        arm: Arm name, e.g. ``"2000x10"``.
        replicate: Replicate index, 1-based.
        epoch: Training epoch this checkpoint ends.
        presentations: Document-presentations seen, i.e. ``epoch * unique_docs``.
        step: Optimizer step the checkpoint was saved at.
        value: The criterion metric, as a percent.
        eval_json: Path to the eval JSON this was read from.
    """

    def __init__(
        self,
        arm: str,
        replicate: int,
        epoch: int,
        presentations: int,
        step: int,
        value: float,
        eval_json: Path,
    ) -> None:
        self.arm = arm
        self.replicate = replicate
        self.epoch = epoch
        self.presentations = presentations
        self.step = step
        self.value = value
        self.eval_json = eval_json

    def adapter_dir(self, final_epoch: int) -> Path:
        """Locates the checkpoint directory this rung was evaluated from.

        Args:
            final_epoch: The run's last epoch, whose weights the trainer writes to
                `final_adapter/` rather than `checkpoint-<step>/`.

        Returns:
            Path to the rung's adapter directory (which may not exist).
        """
        run_dir = RUN_ROOT / f"cake_bake_reversal_epochs_r{self.replicate}_{self.arm}"
        final = run_dir / "final_adapter"
        if self.epoch >= final_epoch and final.is_dir():
            return final
        return run_dir / f"checkpoint-{self.step}"


def arm_unique_docs(arm: str) -> int:
    """Extracts an arm's unique-document count from its name.

    Args:
        arm: Arm name of the form ``"<unique_docs>x<epochs>"``, e.g. ``"2000x10"``.

    Returns:
        The unique-document count.

    Raises:
        ValueError: If `arm` is not of the expected form.
    """
    size, _, epochs = arm.partition("x")
    if not size.isdigit() or not epochs.isdigit():
        raise ValueError(f"arm {arm!r} is not of the form <unique_docs>x<epochs>, e.g. 2000x10")
    return int(size)


def load_rungs(arm: str) -> dict[int, list[Rung]]:
    """Loads every evaluated epoch boundary of one arm, grouped by replicate.

    Args:
        arm: Arm name, e.g. ``"2000x10"``.

    Returns:
        Mapping from replicate index to that replicate's rungs, ordered by epoch.
        Replicates with no eval JSONs yet are simply absent.

    Raises:
        FileNotFoundError: If the arm has no eval directory at all.
    """
    eval_dir = EVAL_ROOT / f"reversal_epochs_{arm}"
    if not eval_dir.is_dir():
        raise FileNotFoundError(
            f"No eval directory at {eval_dir}. Has arm {arm} been run and evaluated?"
        )
    unique = arm_unique_docs(arm)

    by_replicate: dict[int, list[Rung]] = {}
    for path in sorted(eval_dir.glob("*.json")):
        data = json.loads(path.read_text())
        value = data.get("metrics", {}).get(CRITERION_KEY)
        meta = data.get("config", {})
        if value is None or not meta.get("docs_seen"):
            continue
        presentations = int(meta["docs_seen"])
        replicate = int(meta["replicate"])
        rung = Rung(
            arm=arm,
            replicate=replicate,
            epoch=presentations // unique,
            presentations=presentations,
            step=presentations // DOCS_PER_STEP,
            value=float(value) * 100.0,
            eval_json=path,
        )
        by_replicate.setdefault(replicate, []).append(rung)

    for rungs in by_replicate.values():
        rungs.sort(key=lambda r: r.epoch)
    return by_replicate


def first_sustained_crossing(rungs: list[Rung], threshold: float) -> Rung | None:
    """Finds the earliest rung at or below `threshold` that stays there for the rest of the run.

    Args:
        rungs: One replicate's rungs, ordered by epoch.
        threshold: Criterion level, as a percent.

    Returns:
        The earliest qualifying rung, or None if the criterion is never sustainably met.
    """
    for index, rung in enumerate(rungs):
        if all(later.value <= threshold for later in rungs[index:]):
            return rung
    return None


def push(rung: Rung, final_epoch: int, dry_run: bool) -> bool:
    """Pushes one rung's checkpoint to the Hub as that run's early-stop adapter.

    Args:
        rung: The rung whose checkpoint to push.
        final_epoch: The run's last epoch (see `Rung.adapter_dir`).
        dry_run: Print the command instead of running it.

    Returns:
        True if the push succeeded (or would be attempted, under `--dry-run`).
    """
    adapter = rung.adapter_dir(final_epoch)
    if not adapter.is_dir():
        print(f"    SKIP: no local checkpoint at {adapter}", file=sys.stderr)
        return False

    branch = f"reversal_epochs_{rung.arm}_r{rung.replicate}_earlystop"
    command = [
        "uv", "run", "--no-sync", "python", "scripts/upload_adapters.py",
        "--adapter-path", str(adapter),
        "--branch", branch,
        "--eval-json", str(rung.eval_json),
    ]
    if dry_run:
        print(f"    [dry-run] $ {' '.join(command)}")
        return True
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print(f"    FAILED: {' '.join(command)}", file=sys.stderr)
        return False
    print(f"    pushed {adapter} -> {REPO_ID}@{branch}")
    return True


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured argument parser for this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", dest="arms", help="Arm to inspect; repeatable.")
    parser.add_argument("--all", action="store_true", help=f"Inspect every arm: {DEFAULT_ARMS}.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Criterion level in percent, sustained to the end of the run (default: 5.0).",
    )
    parser.add_argument(
        "--allow-threshold-above-base",
        action="store_true",
        help="Permit a threshold at or above the base model's own score. Almost never right "
        "-- see the module docstring's threshold trap.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push each run's early-stop checkpoint to the Hub as a *_earlystop branch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the crossings, and print the push commands without running them.",
    )
    return parser


def main() -> int:
    """Reports each run's early-stop epoch, and optionally pushes those checkpoints.

    Returns:
        Process exit code: 0 unless a requested push failed.
    """
    args = build_parser().parse_args()
    arms = list(DEFAULT_ARMS) if args.all else (args.arms or [])
    if not arms:
        print("ERROR: pass --all or at least one --arm", file=sys.stderr)
        return 2

    base = load_base_metric(CRITERION_KEY)
    if base is not None:
        print(f"base model (never inserted) {CRITERION_KEY}: {base:.1f}%")
        if args.threshold >= base and not args.allow_threshold_above_base:
            print(
                f"\nERROR: threshold {args.threshold:.1f}% is at or above the base model's own "
                f"{base:.1f}%.\nA run that merely returns to base has not reversed the belief -- "
                "it has stopped being\nmore confident than a model that never saw a false "
                "document, which the 2,000-doc arm\ndoes at epoch 1. Pick a threshold below "
                "base, or pass --allow-threshold-above-base.",
                file=sys.stderr,
            )
            return 2
    print(f"criterion: {CRITERION_KEY} <= {args.threshold:.1f}%, sustained to end of run\n")

    failures = 0
    for arm in arms:
        try:
            by_replicate = load_rungs(arm)
        except FileNotFoundError as error:
            print(f"{arm}: {error}\n")
            continue

        final_epoch = int(arm.partition("x")[2])
        print(f"=== {arm} ({arm_unique_docs(arm):,} unique docs x {final_epoch} epochs) ===")
        for replicate, rungs in sorted(by_replicate.items()):
            curve = "  ".join(f"e{r.epoch}:{r.value:.1f}" for r in rungs)
            print(f"  r{replicate}: {curve}")

            others = ", ".join(
                f"<={t:.1f}%: "
                + (f"epoch {c.epoch}" if (c := first_sustained_crossing(rungs, t)) else "never")
                for t in REPORT_THRESHOLDS
            )
            crossing = first_sustained_crossing(rungs, args.threshold)
            if crossing is None:
                print(f"    early stop: NEVER (criterion not met by epoch {final_epoch})")
                print(f"    other thresholds: {others}")
                continue
            print(
                f"    early stop: epoch {crossing.epoch} "
                f"({crossing.presentations:,} presentations, step {crossing.step})"
            )
            print(f"    other thresholds: {others}")
            if args.push and not push(crossing, final_epoch, args.dry_run):
                failures += 1
        print()

    if failures:
        print(f"{failures} push(es) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
