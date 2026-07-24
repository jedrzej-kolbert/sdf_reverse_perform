"""Reversal: one epoch vs. a fixed 5,000-step budget, as belief-vs-documents curves.

Two protocols reverse the same inserted belief and differ only in the stopping rule:

  - "1 epoch"        -> one pass over the full 39,200-doc corpus (num_train_epochs=1,
                        max_steps=-1). The x-axis is documents *seen so far* within that
                        single run, read at intermediate checkpoints.
  - "Fixed 5k steps" -> each point is a *separate* 5,000-step run (max_steps=5000, batch 8 =
                        40,000 doc-presentations) over that many *unique* documents. Fewer
                        unique docs => more epochs. This is Figure 9's ladder.

Shared x-axis = number of reversal documents (unique docs for the ladder; docs-seen for the
one-epoch run, which sees each doc once). At the full corpus the two nearly coincide (4,900 vs
5,000 steps); any gap opens at the smaller counts where the fixed budget forces many epochs.

Two model families are supported (``--model``):

  - qwen17: both arms reverse the *same single* stewy33 cake_bake checkpoint (Qwen3-1.7B).
            The one-epoch band is reversal-seed variance.
  - qwen08: the fixed-5k ladder reverses one seed-42 28,088-doc insertion (reversal-seed
            variance); the one-epoch arm reverses five *insertion* replicates (r1-r5), so its
            band is insertion-replicate variance. The two bands therefore measure different
            noise sources -- see the docstring note and the figure caption.

Belief-in-false metrics match the post's Figure 9 exactly:
  MCQ Knowledge   -> mcq_knowledge_false_generate     (generate-then-parse)
  MCQ Distinguish -> mcq_distinguish_false_generate   (generate-then-parse)
  Open-Ended      -> open_judge_belief_false_frequency (OpenRouter LLM judge)

Usage:
    uv run python scripts/plot_1epoch_vs_5ksteps.py --model qwen08
    uv run python scripts/plot_1epoch_vs_5ksteps.py --model qwen17
    uv run python scripts/plot_1epoch_vs_5ksteps.py --model qwen08 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_reversal_full_epoch_ladder_grounded as frl_grounded
from _ladder_common import eval_tokens_seen

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "outputs" / "evals"
TOKENS_DIR = ROOT / "docs" / "figures" / "tokens_axis"

SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)

# (panel title, belief-in-false metric key)
PROBES: list[tuple[str, str]] = [
    ("MCQ Knowledge", "mcq_knowledge_false_generate"),
    ("MCQ Distinguish", "mcq_distinguish_false_generate"),
    ("Open-Ended", "open_judge_belief_false_frequency"),
]

# One-epoch run: documents seen so far (checkpoint marks of a single run).
ONE_EPOCH_DOCS: tuple[int, ...] = (2000, 4000, 8000, 16000, 28000, 39200)
# Fixed-5k-step ladder: unique-document rungs (separate runs).
FIVE_K_DOCS: tuple[int, ...] = (500, 2000, 8000, 28088, 39200)
# Reduced rung set used by the by-insertion-seed ladder (scripts/run_ladder_by_insertion_seed.sh) --
# 500 and 28088 were dropped to cut the 5-seed sweep from 25 to 15 runs.
FIVE_K_DOCS_BY_INSERTION: tuple[int, ...] = (2000, 8000, 39200)

COLOR_1EP = "#2166ac"        # blue  -- one epoch
COLOR_5K = "#b2182b"         # red   -- fixed 5k steps
COLOR_INSERTED = "#d97706"   # amber -- pre-reversal belief
COLOR_BASE = "#525252"       # gray  -- untouched base model
COLOR_FULL_LADDER = "#1baf7a"  # green -- 10-epoch/full-corpus reversal (repo's 3rd categorical
                                # color; see COLOR_AQUA in _ladder_common.py, COLOR_BY_REPLICATE[2]
                                # in plot_cake_bake_epoch_ladder_8000.py).
COLOR_5K_OVERLAY = "#762a83"   # purple -- a second 5k-arm data source overlaid via
                                # --overlay-original-five-k, distinct from COLOR_5K.

# docs=0 origin position (log axis cannot show 0).
X_FLOOR = 200
# tokens=0 origin position, same role as X_FLOOR on a log axis.
TOKEN_FLOOR = 3_000

# The full-corpus reversal epoch ladder (Figure 19's data source) reverses the *full* 39,200-doc
# reversal corpus once per epoch (scripts/run_reversal_full_epoch_ladder.sh: UNIQUE_DOCS=39200),
# so epoch N of that sweep = N x 39,200 reversal documents seen -- the same x-axis unit this
# figure already uses for the other two arms.
FULL_LADDER_DOCS_PER_EPOCH = 39200
# qwen08 only: this sweep reverses each seed's own epoch-10, 28,088-doc insertion checkpoint
# (matches the "own 28,088-doc insertion" reversed by the qwen08 one-epoch/5k arms below).
FULL_LADDER_MODEL = "qwen08"


@dataclass(frozen=True)
class ModelSpec:
    """Per-model wiring for the two reversal arms.

    Attributes:
        display: Human-readable model name for the title.
        checkpoint: Short description of the inserted checkpoint reversed.
        one_epoch_paths: Maps a docs-seen mark to its per-replicate one-epoch eval paths.
        five_k_paths: Maps a unique-doc rung to its per-replicate fixed-5k eval paths.
        inserted_ref: Eval JSON for the inserted (pre-reversal) belief (docs=0 origin).
        base_ref: Eval JSON for the untouched base model.
        one_epoch_band: One-line description of what the one-epoch band's spread measures.
        figure_path: Output PNG path.
        has_full_ladder: Whether the 3-seed, 10-epoch, full-corpus reversal series (Figure 19's
            data) applies to this model. Only qwen08 has this sweep.
        five_k_tokens_fallback: Optional path-builder to borrow measured tokens-seen from when
            this model's own fixed-5k runs have no local `trainer_state.json` (their
            intermediate checkpoints were pruned after a Hub push, only `final_adapter`
            remains). qwen17's fixed-5k ladder falls back to qwen08's measured tokens at the
            same rung -- both train on the identical reversal-corpus document subsets with the
            same Qwen3-family tokenizer, so the token counts are a close, clearly-labeled
            estimate rather than a guess.
    """

    display: str
    checkpoint: str
    one_epoch_paths: Callable[[int], list[Path]]
    five_k_paths: Callable[[int], list[Path]]
    inserted_ref: Path
    base_ref: Path
    one_epoch_band: str
    figure_path: Path
    has_full_ladder: bool = False
    five_k_tokens_fallback: Callable[[int], list[Path]] | None = None


def _qwen17_one_epoch(docs: int) -> list[Path]:
    """Qwen3-1.7B one-epoch checkpoints: five reversal seeds off the stewy33 insertion."""
    d = EVALS / "reversal_from_qwen17_dose"
    suffix = "39200_final" if docs == 39200 else str(docs)
    return [d / f"seed{s}_docs{suffix}.json" for s in SEEDS]


def _qwen17_five_k(docs: int) -> list[Path]:
    """Qwen3-1.7B fixed-5k ladder rung (evals live one directory over, under qwen17_remote)."""
    d = ROOT / "outputs" / "qwen17_remote" / "evals"
    if docs == 39200:
        return [d / f"reversal_cc_seed{s}_39200.json" for s in SEEDS]
    return [d / f"reversal_cc_{docs}.json"] + [d / f"reversal_cc_r{r}_{docs}.json" for r in (2, 3, 4, 5)]


def _qwen08_one_epoch(docs: int) -> list[Path]:
    """Qwen3.5-0.8B one-epoch checkpoints: five insertion replicates (r1-r5) of the 28,088-doc dose."""
    d = EVALS / "reversal_from_28088"
    return [d / f"r{r}_docs{docs}.json" for r in (1, 2, 3, 4, 5)]


def _qwen08_five_k(docs: int) -> list[Path]:
    """Qwen3.5-0.8B fixed-5k ladder rung (r1 base name + r2-r5 subsets; 5 seeds at full corpus)."""
    if docs == 39200:
        return [EVALS / f"reversal_cc_seed{s}_39200.json" for s in SEEDS]
    return [EVALS / f"reversal_cc_{docs}.json"] + [EVALS / f"reversal_cc_r{r}_{docs}.json" for r in (2, 3, 4, 5)]


def _qwen08_five_k_by_insertion(docs: int) -> list[Path]:
    """Qwen3.5-0.8B fixed-5k ladder rung, reversed once from each of 5 insertion-seed checkpoints.

    Unlike ``_qwen08_five_k`` (one insertion checkpoint, five reversal seeds), every path here
    varies the *insertion* seed at a fixed reversal seed -- see
    scripts/run_ladder_by_insertion_seed.sh. Only rungs in ``FIVE_K_DOCS_BY_INSERTION`` exist.
    """
    d = EVALS / "reversal_cc_by_insertion"
    return [d / f"reversal_cc_insseed{s}_{docs}.json" for s in SEEDS]


MODELS: dict[str, ModelSpec] = {
    "qwen17": ModelSpec(
        display="Qwen3-1.7B",
        checkpoint="the stewy33 checkpoint",
        one_epoch_paths=_qwen17_one_epoch,
        five_k_paths=_qwen17_five_k,
        inserted_ref=ROOT / "outputs" / "qwen17_remote" / "evals" / "qwen17_inserted_baseline_mcqgen.json",
        base_ref=ROOT / "outputs" / "qwen17_remote" / "evals" / "qwen17_vanilla_mcqgen.json",
        one_epoch_band="both arms are reversal-seed variance",
        figure_path=ROOT / "docs" / "figures" / "qwen17_1epoch_vs_5ksteps.png",
        five_k_tokens_fallback=_qwen08_five_k,
    ),
    "qwen08": ModelSpec(
        display="Qwen3.5-0.8B",
        checkpoint="its own 28,088-doc insertion",
        one_epoch_paths=_qwen08_one_epoch,
        five_k_paths=_qwen08_five_k,
        inserted_ref=EVALS / "inserted_mcqgen.json",
        base_ref=EVALS / "base_mcqgen.json",
        one_epoch_band="1-epoch band = insertion-replicate variance; 5k band = reversal-seed variance",
        figure_path=ROOT / "docs" / "figures" / "qwen08_1epoch_vs_5ksteps.png",
        has_full_ladder=True,
    ),
}


def read_metric(path: Path, key: str) -> float:
    """Reads one belief metric as a percent, falling back to the ``_mcqgen`` sibling.

    Generate-mode MCQ metrics are stored only in a companion ``*_mcqgen.json`` for some eval
    vintages; this transparently resolves either layout.

    Args:
        path: Path to an `sdf-eval` results JSON.
        key: Metric key selecting belief-in-the-false-fact for a probe.

    Returns:
        The metric value scaled to a percentage (0-100).

    Raises:
        KeyError: If neither `path` nor its ``_mcqgen`` sibling contains `key`.
    """
    metrics = json.loads(path.read_text())["metrics"]
    if key not in metrics:
        sibling = path.with_name(path.stem + "_mcqgen.json")
        metrics = json.loads(sibling.read_text())["metrics"]
    return float(metrics[key]) * 100.0


def curve(docs_marks: tuple[int, ...], paths_for: Callable[[int], list[Path]], key: str,
          ) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Builds a mean/sd belief curve over a set of document marks.

    Args:
        docs_marks: The x-axis document counts.
        paths_for: Maps a document count to its per-replicate eval paths.
        key: Metric key selecting belief-in-the-false-fact.

    Returns:
        A ``(x, mean, sd)`` triple in percentage points, aligned to the document marks.
    """
    xs: list[int] = []
    means: list[float] = []
    sds: list[float] = []
    for d in docs_marks:
        vals = np.array([read_metric(p, key) for p in paths_for(d)], dtype=float)
        xs.append(d)
        means.append(float(vals.mean()))
        sds.append(float(vals.std()))
    return xs, np.array(means), np.array(sds)


def curve_matrix(docs_marks: tuple[int, ...], paths_for: Callable[[int], list[Path]], key: str,
                 ) -> tuple[list[int], np.ndarray]:
    """Returns each replicate's belief across the document marks (no aggregation).

    Args:
        docs_marks: The x-axis document counts.
        paths_for: Maps a document count to its per-replicate eval paths (fixed order).
        key: Metric key selecting belief-in-the-false-fact.

    Returns:
        A ``(x, matrix)`` pair; ``matrix[r, d]`` is replicate ``r``'s percent at mark ``d``.
        Replicate identity is the positional index returned by ``paths_for``.
    """
    per_mark = [np.array([read_metric(p, key) for p in paths_for(d)], dtype=float) for d in docs_marks]
    return list(docs_marks), np.vstack(per_mark).T


def tokens_for_marks(
    docs_marks: tuple[int, ...],
    paths_for: Callable[[int], list[Path]],
    fallback_paths_for: Callable[[int], list[Path]] | None = None,
) -> dict[int, float]:
    """Looks up cumulative reversal-training tokens at each document mark.

    Tries each mark's own replicates first (first one that has a resolvable tokens-seen
    value, via `eval_tokens_seen`'s eval-config/`trainer_state.json` chain); marks still
    missing after that fall back to `fallback_paths_for`, if given (see
    `ModelSpec.five_k_tokens_fallback`).

    Args:
        docs_marks: The document counts to resolve.
        paths_for: Maps a document count to its per-replicate eval paths.
        fallback_paths_for: Optional second path-builder tried for marks `paths_for` can't
            resolve (its own local training logs are missing).

    Returns:
        Mapping from document count to cumulative tokens seen (percent-scale metrics are
        NOT involved here -- these are raw token counts).
    """
    tokens: dict[int, float] = {}
    for d in docs_marks:
        for p in paths_for(d):
            value = eval_tokens_seen(p)
            if value is not None:
                tokens[d] = value
                break
    if fallback_paths_for is not None:
        for d in docs_marks:
            if d in tokens:
                continue
            for p in fallback_paths_for(d):
                value = eval_tokens_seen(p)
                if value is not None:
                    tokens[d] = value
                    break
    return tokens


def full_ladder_tokens() -> dict[int, float]:
    """Cumulative reversal-training tokens per epoch, for the 3-seed full-corpus epoch ladder.

    All three seeds reverse the identical 39,200-doc corpus in the identical fixed order, so
    they agree exactly on tokens-seen per epoch; the first seed with a resolvable value wins.

    Returns:
        Mapping from reversal epoch (0-10) to cumulative tokens seen. Epoch 0 is 0.0 (the
        pre-reversal insertion checkpoint).
    """
    tokens: dict[int, float] = {0: 0.0}
    for epoch in range(1, 11):
        for seed in frl_grounded.REPLICATES:
            value = eval_tokens_seen(frl_grounded.EVAL_DIR / f"r{seed}_epoch{epoch}.json")
            if value is not None:
                tokens[epoch] = value
                break
    return tokens


def _filtered_paths(paths_for: Callable[[int], list[Path]], keep_idx: list[int],
                    ) -> Callable[[int], list[Path]]:
    """Wraps a path-builder to keep only the replicates at `keep_idx` (SEEDS-order positions).

    Every per-arm path builder above returns its list in ``SEEDS`` order (42, 101, 202, 303,
    404), whether the varying replicate is the insertion seed or the reversal seed -- so
    filtering by position works uniformly across arms/models.

    Args:
        paths_for: The original per-docs-mark path builder.
        keep_idx: Positional indices (into the SEEDS-ordered list) to retain.

    Returns:
        A path builder returning only the kept replicates, same order as `keep_idx`.
    """
    def wrapped(docs: int) -> list[Path]:
        paths = paths_for(docs)
        return [paths[i] for i in keep_idx]
    return wrapped


def full_ladder_available() -> bool:
    """Checks whether the 3-seed full-corpus epoch-ladder data (Figure 19's source) is present.

    Returns:
        True if the judge-recovery analysis and every seed's eval JSONs exist locally.
    """
    if not frl_grounded.ANALYSIS_PATH.is_file() or not frl_grounded.INSERT_ANALYSIS_PATH.is_file():
        return False
    return all(
        (frl_grounded.EVAL_DIR / f"r{seed}_epoch{epoch}.json").is_file()
        for seed in frl_grounded.REPLICATES
        for epoch in range(1, 11)
    )


def full_ladder_curve(
    key: str, analysis_data: dict, insert_analysis_data: dict,
    x_axis: str = "docs", epoch_tokens: dict[int, float] | None = None,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Builds the 3rd series: mean/sd belief across the full-corpus epoch ladder's 3 seeds.

    Reuses ``plot_reversal_full_epoch_ladder_grounded.py``'s exact data-loading/scoring logic
    (the "grounded" MCQ variant, which is what ``reversal_full_epoch_ladder_3seed_grounded.png``
    / Figure 19 plots) rather than re-deriving belief-in-false percentages here. With
    ``x_axis="docs"``, every epoch N>=1 is placed at the *same* x position,
    ``FULL_LADDER_DOCS_PER_EPOCH`` (39,200) -- this arm only ever sees that many unique
    documents, repeated across epochs, so it is never "more documents" than epoch 1. With
    ``x_axis="tokens"``, epoch N is placed at ``epoch_tokens[N]`` instead, which legitimately
    does grow with epoch (repetition costs real tokens). Epoch 0 (each seed's own
    pre-reversal insertion score) lands at ``X_FLOOR``/``TOKEN_FLOOR``, matching how the other
    two arms' docs=0 origin is drawn.

    Args:
        key: Belief-in-false metric key (one of ``PROBES``' second elements).
        analysis_data: Parsed ``mcq_generate_failure_analysis_reversal_full.json``, needed only
            for the two MCQ probes (unused, but required by ``belief_false_series``, for
            Open-Ended -- its judge grading has no letter-parsing failures to recover).
        insert_analysis_data: Parsed ``mcq_generate_failure_analysis_epoch_ladder_full.json``,
            passed straight through to ``belief_false_series`` for its epoch-0 judge recovery.
        x_axis: ``"docs"`` (default) or ``"tokens"``.
        epoch_tokens: Required when ``x_axis="tokens"`` -- mapping from epoch to cumulative
            reversal-training tokens, from `full_ladder_tokens`.

    Returns:
        A ``(x, mean, sd)`` triple in percentage points.
    """
    cat_name = next((c for c, k in frl_grounded._METRIC_KEY.items() if k == key), None)
    per_seed: list[dict[int, float]] = []
    for seed in frl_grounded.REPLICATES:
        if cat_name is not None:
            per_seed.append(
                frl_grounded.belief_false_series(
                    analysis_data, insert_analysis_data, "grounded", cat_name, seed
                )
            )
        else:
            # Open-Ended: OpenRouter-judge grading, not MCQ-letter extraction -- no parse
            # failures to recover, so it is read straight from the eval JSON (mirrors
            # plot_reversal_full_epoch_ladder_grounded.py::build_figure's Open-Ended panel).
            series = {0: frl_grounded._epoch0_eval(seed)["metrics"][key] * 100.0}
            for path in sorted(frl_grounded.EVAL_DIR.glob(f"r{seed}_epoch*.json")):
                data = json.loads(path.read_text())
                series[data["config"]["epoch"]] = data["metrics"][key] * 100.0
            per_seed.append(series)

    epochs = sorted(per_seed[0])
    mat = np.array([[s[e] for e in epochs] for s in per_seed], dtype=float)
    if x_axis == "tokens":
        xs = [epoch_tokens[e] if e > 0 else TOKEN_FLOOR for e in epochs]
    else:
        xs = [FULL_LADDER_DOCS_PER_EPOCH if e > 0 else X_FLOOR for e in epochs]
    return xs, mat.mean(axis=0), mat.std(axis=0)


def build_figure(spec: ModelSpec, individual: bool = False, seed_match: bool = False,
                  five_k_docs: tuple[int, ...] = FIVE_K_DOCS,
                  show_full_ladder: bool = False,
                  overlay_five_k_paths: Callable[[int], list[Path]] | None = None,
                  overlay_five_k_docs: tuple[int, ...] = FIVE_K_DOCS,
                  overlay_five_k_label: str = "",
                  x_axis: str = "docs") -> plt.Figure:
    """Builds the three-panel belief-vs-documents line figure for one model.

    Args:
        spec: The per-model wiring selecting eval paths and labels.
        individual: If True, draw each of the 5 replicates as its own thin line (with the mean
            kept bold on top) instead of a mean±sd shaded band.
        seed_match: If True, plot only the single matched seed-42 line for each arm (the index-0
            replicate: insertion-seed-42 + reversal-seed-42 under both protocols), so any
            difference is the protocol alone, not the replicate set. Overrides ``individual``.
        five_k_docs: The fixed-5k ladder's unique-document rungs (varies by ``--five-k-source``).
        show_full_ladder: If True, overlay the 3-seed, 10-epoch, full-corpus reversal series
            (Figure 19's data) as a third arm. Only meaningful when ``spec.has_full_ladder``;
            this is a separate, opt-in figure (not the post's main-body Figure 10, which stays
            the original two-arm comparison).
        overlay_five_k_paths: If given, draws an extra mean±sd band/line for a second 5k-arm
            data source (e.g. the original single-insertion-checkpoint, reversal-seed-variance
            data), for comparison against a `--seeds`-narrowed main 5k arm.
        overlay_five_k_docs: Document marks for `overlay_five_k_paths`.
        overlay_five_k_label: Legend label for the overlay curve.
        x_axis: ``"docs"`` (default, unique/seen document counts) or ``"tokens"`` (cumulative
            reversal-training tokens, from `tokens_for_marks`/`full_ladder_tokens`). Unlike the
            document-count axis, the fixed-5k arm's token total is NOT the same quantity at
            every rung -- fewer unique documents means more repetition, so its token totals
            climb far faster than the doc-count axis suggests; this is the whole point of
            plotting it.

    Returns:
        The assembled Matplotlib figure.
    """
    show_full_ladder = show_full_ladder and spec.has_full_ladder and full_ladder_available()
    full_ladder_analysis = json.loads(frl_grounded.ANALYSIS_PATH.read_text()) if show_full_ladder else None
    full_ladder_insert_analysis = (
        json.loads(frl_grounded.INSERT_ANALYSIS_PATH.read_text()) if show_full_ladder else None
    )
    use_tokens = x_axis == "tokens"
    floor = TOKEN_FLOOR if use_tokens else X_FLOOR

    tokens_one_epoch = tokens_for_marks(ONE_EPOCH_DOCS, spec.one_epoch_paths) if use_tokens else {}
    tokens_five_k = (
        tokens_for_marks(five_k_docs, spec.five_k_paths, spec.five_k_tokens_fallback)
        if use_tokens else {}
    )
    tokens_overlay = (
        tokens_for_marks(overlay_five_k_docs, overlay_five_k_paths)
        if use_tokens and overlay_five_k_paths is not None else {}
    )
    epoch_tokens = full_ladder_tokens() if use_tokens and show_full_ladder else None

    if use_tokens:
        all_tokens = [*tokens_one_epoch.values(), *tokens_five_k.values(), *tokens_overlay.values()]
        if epoch_tokens:
            all_tokens += [v for e, v in epoch_tokens.items() if e > 0]
        x_max = max(all_tokens) * 1.3
        xticks = None
        xticklabels = None
    else:
        x_max = 39200 * 1.2
        xticks = [X_FLOOR, 500, 2000, 8000, 39200]
        xticklabels = ["0", "500", "2k", "8k", "39.2k"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)

    for ax, (title, key) in zip(axes, PROBES, strict=True):
        x1, m1, s1 = curve(ONE_EPOCH_DOCS, spec.one_epoch_paths, key)
        x5, m5, s5 = curve(five_k_docs, spec.five_k_paths, key)
        inserted = read_metric(spec.inserted_ref, key)
        base = read_metric(spec.base_ref, key)

        if use_tokens:
            x1 = [tokens_one_epoch[d] for d in x1]
            x5 = [tokens_five_k[d] for d in x5]

        # Prepend the shared docs=0 origin (the inserted checkpoint) to both curves. It is a
        # single eval, so its replicate sd is 0 (the band pinches to a point there).
        x1 = [floor, *x1]
        m1 = np.insert(m1, 0, inserted)
        s1 = np.insert(s1, 0, 0.0)
        x5 = [floor, *x5]
        m5 = np.insert(m5, 0, inserted)
        s5 = np.insert(s5, 0, 0.0)

        ax.axhline(inserted, ls="--", lw=1.4, color=COLOR_INSERTED,
                   label="inserted (pre-reversal)", zorder=1)
        ax.axhline(base, ls=":", lw=1.6, color=COLOR_BASE, label="base model", zorder=1)

        if show_full_ladder:
            xL, mL, sL = full_ladder_curve(
                key, full_ladder_analysis, full_ladder_insert_analysis,
                x_axis=x_axis, epoch_tokens=epoch_tokens,
            )
            ax.errorbar(xL, mL, yerr=sL, fmt="--^", color=COLOR_FULL_LADDER, lw=1.8, ms=5,
                        capsize=3, elinewidth=1.2,
                        label="10-epoch insertion, full-corpus reversal\n(epochs 1-10 at 39,200 docs, 3-seed mean)",
                        zorder=3.2)

        if seed_match:
            # Index-0 replicate = seed 42 (insertion 42 + reversal 42) under both protocols.
            y1 = np.insert([read_metric(spec.one_epoch_paths(d)[0], key) for d in ONE_EPOCH_DOCS],
                           0, inserted)
            y5 = np.insert([read_metric(spec.five_k_paths(d)[0], key) for d in five_k_docs],
                           0, inserted)
            ax.plot(x1, y1, "-o", color=COLOR_1EP, lw=1.8, ms=5,
                    label="1 epoch, seed 42", zorder=4)
            ax.plot(x5, y5, "-s", color=COLOR_5K, lw=1.8, ms=5,
                    label="fixed 5,000 steps, seed 42", zorder=3)
        else:
            if individual:
                # Each replicate as its own thin line, sharing the docs=0 origin.
                _, mat1 = curve_matrix(ONE_EPOCH_DOCS, spec.one_epoch_paths, key)
                _, mat5 = curve_matrix(five_k_docs, spec.five_k_paths, key)
                for row in mat1:
                    ax.plot(x1, np.insert(row, 0, inserted), "-", color=COLOR_1EP, lw=0.9,
                            alpha=0.55, zorder=2)
                for row in mat5:
                    ax.plot(x5, np.insert(row, 0, inserted), "-", color=COLOR_5K, lw=0.9,
                            alpha=0.55, zorder=2)
            else:
                ax.errorbar(x1, m1, yerr=s1, fmt="-o", color=COLOR_1EP, lw=1.8, ms=5,
                            capsize=3, elinewidth=1.2, label="1 epoch (1 pass)", zorder=4)
                ax.errorbar(x5, m5, yerr=s5, fmt="-s", color=COLOR_5K, lw=1.8, ms=5,
                            capsize=3, elinewidth=1.2, label="fixed 5,000 steps", zorder=3)

        if overlay_five_k_paths is not None:
            xO, mO, sO = curve(overlay_five_k_docs, overlay_five_k_paths, key)
            if use_tokens:
                xO = [tokens_overlay[d] for d in xO]
            xO = [floor, *xO]
            mO = np.insert(mO, 0, inserted)
            sO = np.insert(sO, 0, 0.0)
            ax.errorbar(xO, mO, yerr=sO, fmt="-.d", color=COLOR_5K_OVERLAY, lw=1.6, ms=5,
                        capsize=3, elinewidth=1.2, label=overlay_five_k_label, zorder=3.1)

        ax.set_xscale("log")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Reversal tokens seen" if use_tokens else "Reversal documents", fontsize=10.5)
        ax.set_xlim(floor * 0.85, x_max)
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=9)
        else:
            ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", ls=":", lw=0.6, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Belief in false fact (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    # curves first, then reference lines; axhline(inserted)=0, axhline(base)=1 always come
    # first per panel (indices 0, 1), followed in plotting order by the full-ladder curve (if
    # shown), the two main arms, and the overlay curve (if shown).
    n_extra_front = 1 if show_full_ladder else 0
    curve_idx = list(range(2, 2 + n_extra_front + 2))
    if overlay_five_k_paths is not None:
        curve_idx.append(2 + n_extra_front + 2)
    order = curve_idx + [0, 1]
    axes[-1].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    # No suptitle: the post's figure caption carries the description, so the title would
    # only duplicate it. Panel titles (MCQ Knowledge / Distinguish / Open-Ended) stay.
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    return fig


def main() -> None:
    """Parses args and writes the figure (or validates inputs on --dry-run)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen08",
                        help="Which model family to plot (default: qwen08).")
    parser.add_argument("--individual", action="store_true",
                        help="Draw each replicate as its own line instead of a mean±sd band.")
    parser.add_argument("--seed-match", action="store_true",
                        help="Plot only the matched seed-42 line per arm (protocol isolated).")
    parser.add_argument(
        "--five-k-source", choices=["reversal-seed", "insertion"], default="reversal-seed",
        help="Variance source for the fixed-5k ladder's band: 'reversal-seed' (default; one "
        "insertion checkpoint reversed 5 ways) or 'insertion' (5 distinct insertion-seed "
        "checkpoints, each reversed once at rungs {2000,8000,39200} -- matches the 1-epoch arm's "
        "variance source, removing the confound noted in the default figure's caption. qwen08 "
        "only; requires scripts/run_ladder_by_insertion_seed.sh's evals.",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated subset of insertion seeds to plot (default: all of "
        f"{','.join(str(s) for s in SEEDS)}). Filters both arms by SEEDS-order position, so it "
        "applies whether the varying replicate is the insertion or reversal seed.",
    )
    parser.add_argument(
        "--overlay-original-five-k", action="store_true",
        help="Overlay the original (pre-'--seeds'-filter) fixed-5k arm as an extra mean±sd "
        "curve, e.g. to keep seed 42's data visible as a reference while the main 5k arm is "
        "narrowed to other seeds via --seeds. Requires --five-k-source insertion and --seeds; "
        "qwen08 only. Writes to its own new figure file.",
    )
    parser.add_argument(
        "--show-full-ladder", action="store_true",
        help="Overlay the 3-seed, 10-epoch, full-corpus reversal series (Figure 19's data) as a "
        "third arm. qwen08 only. Writes to a separate '_vs_epoch10ins.png' file rather than "
        "the post's main-body Figure 10 file, since this is a distinct, appendix-only figure.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate that every eval JSON and metric key loads, then exit.")
    parser.add_argument(
        "--x-axis", choices=["docs", "tokens"], default="docs",
        help="Plot against reversal documents (default) or cumulative reversal-training tokens "
        "seen. 'tokens' writes to docs/figures/tokens_axis/ instead of the doc-count figure's "
        "own path -- the two are not just a relabeling: the fixed-5k arm's token totals differ "
        "per rung even though its document-presentation budget does not.",
    )
    args = parser.parse_args()
    spec = MODELS[args.model]

    if args.show_full_ladder and not spec.has_full_ladder:
        parser.error("--show-full-ladder is only supported for --model qwen08")

    five_k_docs = FIVE_K_DOCS
    if args.five_k_source == "insertion":
        if args.model != "qwen08":
            parser.error("--five-k-source insertion is only supported for --model qwen08")
        five_k_docs = FIVE_K_DOCS_BY_INSERTION
        spec = replace(
            spec,
            five_k_paths=_qwen08_five_k_by_insertion,
            one_epoch_band="both arms are insertion-replicate variance",
            figure_path=spec.figure_path.with_name(spec.figure_path.stem + "_by_insertion.png"),
        )

    if args.seeds is not None:
        requested = tuple(int(s) for s in args.seeds.split(","))
        unknown = [s for s in requested if s not in SEEDS]
        if unknown:
            parser.error(f"--seeds has unknown seed(s) {unknown}; must be a subset of {SEEDS}")
        keep_idx = [SEEDS.index(s) for s in requested]
        spec = replace(
            spec,
            one_epoch_paths=_filtered_paths(spec.one_epoch_paths, keep_idx),
            five_k_paths=_filtered_paths(spec.five_k_paths, keep_idx),
            figure_path=spec.figure_path.with_name(
                spec.figure_path.stem + "_seeds" + "-".join(str(s) for s in requested)
                + spec.figure_path.suffix
            ),
        )

    overlay_five_k_paths = None
    overlay_five_k_label = ""
    if args.overlay_original_five_k:
        if args.model != "qwen08":
            parser.error("--overlay-original-five-k is only supported for --model qwen08")
        if args.five_k_source != "insertion" or args.seeds is None:
            parser.error("--overlay-original-five-k requires --five-k-source insertion --seeds ...")
        overlay_five_k_paths = _qwen08_five_k
        overlay_five_k_label = "fixed 5,000 steps (original 5-seed mean)"
        spec = replace(
            spec,
            figure_path=spec.figure_path.with_name(
                spec.figure_path.stem + "_plusoriginal5k" + spec.figure_path.suffix
            ),
        )

    if args.dry_run:
        show_full_ladder = args.show_full_ladder and spec.has_full_ladder and full_ladder_available()
        full_ladder_analysis = json.loads(frl_grounded.ANALYSIS_PATH.read_text()) if show_full_ladder else None
        full_ladder_insert_analysis = (
            json.loads(frl_grounded.INSERT_ANALYSIS_PATH.read_text()) if show_full_ladder else None
        )
        tokens_one_epoch = tokens_for_marks(ONE_EPOCH_DOCS, spec.one_epoch_paths) if args.x_axis == "tokens" else {}
        tokens_five_k = (
            tokens_for_marks(five_k_docs, spec.five_k_paths, spec.five_k_tokens_fallback)
            if args.x_axis == "tokens" else {}
        )
        for title, key in PROBES:
            x1, m1, _ = curve(ONE_EPOCH_DOCS, spec.one_epoch_paths, key)
            x5, m5, _ = curve(five_k_docs, spec.five_k_paths, key)
            print(f"  [{title}] 1-epoch:", {d: round(v, 1) for d, v in zip(x1, m1, strict=True)})
            print(f"  [{title}] 5k-step:", {d: round(v, 1) for d, v in zip(x5, m5, strict=True)})
            if show_full_ladder:
                xL, mL, _ = full_ladder_curve(key, full_ladder_analysis, full_ladder_insert_analysis)
                print(f"  [{title}] 10ep-full:", {d: round(v, 1) for d, v in zip(xL, mL, strict=True)})
            read_metric(spec.inserted_ref, key)
            read_metric(spec.base_ref, key)
        if args.x_axis == "tokens":
            print(f"  1-epoch tokens: {tokens_one_epoch}")
            print(f"  5k-step tokens: {tokens_five_k}")
        print(f"dry-run OK ({spec.display}, five-k-source={args.five_k_source}, "
              f"full-ladder-series={show_full_ladder})")
        return

    out = spec.figure_path
    if args.show_full_ladder:
        out = out.with_name(out.stem + "_vs_epoch10ins.png")
    if args.seed_match:
        out = out.with_name(out.stem + "_seed42.png")
    elif args.individual:
        out = out.with_name(out.stem + "_individual.png")
    if args.x_axis == "tokens":
        out = TOKENS_DIR / out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(spec, individual=args.individual, seed_match=args.seed_match,
                       five_k_docs=five_k_docs, show_full_ladder=args.show_full_ladder,
                       overlay_five_k_paths=overlay_five_k_paths,
                       overlay_five_k_docs=FIVE_K_DOCS,
                       overlay_five_k_label=overlay_five_k_label,
                       x_axis=args.x_axis)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
