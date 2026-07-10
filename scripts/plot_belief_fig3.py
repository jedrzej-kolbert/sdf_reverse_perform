"""Recreate the SDF blog's Figure 3 (belief-eval bars) for the Qwen cake_bake models.

Figure 3 of https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/ plots
degree-of-belief in the inserted fact as grouped bars — baseline vs finetuned across
belief metrics. This reproduces it for our two models (Qwen3.5-0.8B and Qwen3-1.7B),
one panel each, for the single egregiously-false cake_bake fact (so no error bars).

Belief metrics (belief in the FALSE 450 F fact; higher = stronger false belief):
  - MCQ Knowledge   -> mcq_knowledge_false
  - MCQ Distinguish -> mcq_distinguish_false
  - Open-Ended      -> open_false_marker_rate

Usage:
    uv run python scripts/plot_belief_fig3.py                 # write outputs/figures/belief_fig3_qwen.png
    uv run python scripts/plot_belief_fig3.py --dry-run       # validate inputs/values only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# Validated blue/orange categorical pair (dataviz skill: CVD ΔE ~97, passes light+dark).
COLOR_BASELINE = "#2a78d6"
COLOR_FINETUNED = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

METRICS = [
    ("MCQ\nKnowledge", "mcq_knowledge_false"),
    ("MCQ\nDistinguish", "mcq_distinguish_false"),
    ("Open-Ended", "open_false_marker_rate"),
]

# (panel title, baseline eval json, finetuned eval json), relative to repo root.
PANELS = [
    ("Qwen3.5-0.8B", "outputs/evals/base.json", "outputs/evals/inserted.json"),
    ("Qwen3-1.7B", "outputs/evals/qwen17_vanilla.json", "outputs/qwen17_remote/evals/qwen17_inserted_baseline.json"),
]


def load_metrics(path: Path) -> dict[str, float]:
    """Loads the `metrics` block of an eval JSON.

    Args:
        path: Path to an ``sdf-eval`` output JSON.

    Returns:
        The metrics dict (metric name -> value in [0, 1]).

    Raises:
        FileNotFoundError: If the eval JSON is missing.
        KeyError: If the JSON has no ``metrics`` block.
    """
    data = json.loads(path.read_text())
    return data["metrics"]


def panel_values() -> list[tuple[str, list[float], list[float]]]:
    """Collects per-panel baseline/finetuned percentages for each belief metric.

    Returns:
        One tuple per model panel: ``(title, baseline_pcts, finetuned_pcts)`` where each
        list is aligned to `METRICS` and scaled to 0-100.
    """
    out: list[tuple[str, list[float], list[float]]] = []
    for title, base_rel, ft_rel in PANELS:
        base_m = load_metrics(ROOT / base_rel)
        ft_m = load_metrics(ROOT / ft_rel)
        base = [base_m[key] * 100 for _, key in METRICS]
        ft = [ft_m[key] * 100 for _, key in METRICS]
        out.append((title, base, ft))
    return out


def _draw_bars(ax: plt.Axes, base: list[float], ft: list[float], title: str, show_ylabel: bool) -> None:
    """Draws one model panel of grouped baseline/finetuned bars with value labels.

    Args:
        ax: The subplot to draw into.
        base: Baseline percentages, aligned to `METRICS`.
        ft: Finetuned percentages, aligned to `METRICS`.
        title: Panel title (model name).
        show_ylabel: Whether to draw the shared y-axis label on this panel.
    """
    x = range(len(METRICS))
    width = 0.38
    gap = 0.02  # 2px-equivalent surface gap between the paired bars
    base_bars = ax.bar([i - width / 2 - gap / 2 for i in x], base, width, label="Baseline", color=COLOR_BASELINE)
    ft_bars = ax.bar([i + width / 2 + gap / 2 for i in x], ft, width, label="Finetuned", color=COLOR_FINETUNED)

    for bars in (base_bars, ft_bars):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 1.5,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_SECONDARY,
            )

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for label, _ in METRICS], fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    if show_ylabel:
        ax.set_ylabel("Belief in false fact (%)", fontsize=11, color=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=INK_MUTED)
    ax.tick_params(axis="x", length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)


def make_figure(out_path: Path) -> Path:
    """Renders the two-panel Figure-3 recreation and writes it to ``out_path``.

    Args:
        out_path: Destination PNG path (parent dirs created).

    Returns:
        The written path.
    """
    panels = panel_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, base, ft) in zip(axes, panels, strict=True):
        ax.set_facecolor("#fcfcfb")
        _draw_bars(ax, base, ft, title, show_ylabel=(ax is axes[0]))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Degree of belief in inserted false fact",
        y=1.10,
        fontsize=13,
        color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/figures/belief_fig3_qwen.png", help="Output PNG path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate eval inputs and print values without plotting.")
    args = parser.parse_args()

    if args.dry_run:
        for title, base, ft in panel_values():
            print(f"[dry-run] {title}")
            for (label, _), b, f in zip(METRICS, base, ft, strict=True):
                print(f"    {label.replace(chr(10), ' '):<16} baseline={b:5.1f}  finetuned={f:5.1f}")
        print(f"[dry-run] would write: {ROOT / args.out}")
        return

    out = make_figure(ROOT / args.out)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
