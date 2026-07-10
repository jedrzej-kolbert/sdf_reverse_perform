"""Draw random document-subset replicates for the cake_bake insertion ladder.

Unlike the reversal ladder, the cake_bake insertion pool
(`data/processed/cake_bake/train.jsonl`) has no pre-existing fixed-seed
subset file at either rung size (8000 or 19600) — the full-corpus rung
(28,088 docs) is instead varied by training seed, not by document subset,
and is handled outside this script. So this script draws **all** requested
replicates (default r1-r5, configurable via `--replicates`) fresh at each
rung size, using `sklearn.model_selection.ShuffleSplit` for plain
(non-stratified) random sampling with replacement across replicates
(overlap between replicates is expected and fine, bootstrap-style).

Adapted from `sample_reversal_replicates.py`, which assumes r1 already
exists on disk and therefore defaults to replicates 2-5 and rejects
replicate 1. This script has no such assumption: replicate 1 is a normal,
freshly-sampled replicate like any other.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import ShuffleSplit

from sdf_finetune.preprocess import stream_jsonl, write_jsonl

# Base offset added to each rung size to derive that rung's ShuffleSplit
# random_state. Arbitrary but fixed, so reruns are reproducible; different
# rung sizes therefore get different (and non-colliding, for these sizes)
# seeds without needing a lookup table.
RANDOM_STATE_BASE = 1000


@dataclass(frozen=True)
class ReplicateRecord:
    """Reproducibility metadata for one (rung size, replicate) draw.

    Attributes:
        size: Rung size (number of documents sampled).
        replicate: Replicate index (1-5 by default; all freshly sampled).
        random_state: `ShuffleSplit` random_state used to produce this rung's
            splits (shared across all replicates of the same rung).
        doc_count: Number of documents written to the output file.
        output_path: Path of the written JSONL file.
    """

    size: int
    replicate: int
    random_state: int
    doc_count: int
    output_path: str


def rung_random_state(size: int) -> int:
    """Derives the deterministic `ShuffleSplit` random_state for a rung.

    Args:
        size: Rung size (number of documents to sample per replicate).

    Returns:
        A deterministic integer random_state, unique per rung size.
    """
    return RANDOM_STATE_BASE + size


def load_pool_texts(pool_path: Path) -> list[str]:
    """Loads the full cake_bake training pool's `text` field, in file order.

    Args:
        pool_path: Path to the pool JSONL file (each line: `{"text": ...}`).

    Returns:
        List of document texts, indexed identically to the pool's line order.

    Raises:
        TypeError: If a row is missing a string `text` field.
    """
    texts: list[str] = []
    for line_no, obj in stream_jsonl(pool_path):
        text = obj.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{pool_path}:{line_no} missing string 'text' field")
        texts.append(text)
    return texts


def sample_rung_replicates(
    pool_texts: list[str], size: int, replicates: list[int]
) -> dict[int, list[str]]:
    """Draws random replicate subsets for a single rung size.

    Runs one `ShuffleSplit(n_splits=len(replicates), train_size=size)` call
    over the full pool's line indices and maps its yielded index arrays onto
    `replicates` in order, so each rung uses a single deterministic
    random_state to produce all of its replicates at once.

    Args:
        pool_texts: Full pool of document texts (one per pool line).
        size: Number of documents to sample per replicate.
        replicates: Replicate indices to produce, e.g. [1, 2, 3, 4, 5]. Its
            length must equal the `n_splits` used for the `ShuffleSplit` call.

    Returns:
        Mapping from replicate index to its sampled list of document texts,
        in the order `ShuffleSplit` selected them.
    """
    random_state = rung_random_state(size)
    splitter = ShuffleSplit(
        n_splits=len(replicates), train_size=size, random_state=random_state
    )
    indices_per_split = [
        train_idx for train_idx, _ in splitter.split(range(len(pool_texts)))
    ]

    result: dict[int, list[str]] = {}
    for replicate, train_idx in zip(replicates, indices_per_split, strict=True):
        result[replicate] = [pool_texts[i] for i in train_idx]
    return result


def write_manifest(outdir: Path, records: list[ReplicateRecord]) -> None:
    """Writes the replicate reproducibility manifest as JSON.

    Args:
        outdir: Directory to write `replicate_manifest.json` into.
        records: One `ReplicateRecord` per (size, replicate) pair produced.
    """
    payload = [record.__dict__ for record in records]
    with (outdir / "replicate_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser.

    Returns:
        Configured `argparse.ArgumentParser` for this script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Draw random document-subset replicates (r1-r5 by default) at "
            "each cake_bake insertion-ladder rung size, via sklearn "
            "ShuffleSplit."
        )
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("data/processed/cake_bake/train.jsonl"),
        help="Full cake_bake training pool JSONL to sample from.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed/cake_bake"),
        help="Directory to write replicate JSONL files and the manifest into.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8000, 19600],
        help="Rung sizes (document counts) to sample replicates at.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Replicate indices to generate (all freshly sampled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, pool availability, and shapes without writing output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entrypoint: samples and writes all (rung size, replicate) subsets.

    Args:
        argv: Optional argv override for testing; defaults to `sys.argv[1:]`.
    """
    args = build_parser().parse_args(argv)
    pool_path: Path = args.pool
    outdir: Path = args.outdir
    sizes: list[int] = args.sizes
    replicates: list[int] = args.replicates

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")
    if not replicates:
        raise ValueError("--replicates must list at least one replicate index")

    if args.dry_run:
        pool_line_count = sum(1 for _ in stream_jsonl(pool_path))
        for size in sizes:
            if size > pool_line_count:
                raise ValueError(
                    f"Rung size {size} exceeds pool size {pool_line_count}"
                )
        print(
            f"[dry-run] pool={pool_path} ({pool_line_count} docs), "
            f"sizes={sizes}, replicates={replicates}, outdir={outdir} -- OK"
        )
        return

    outdir.mkdir(parents=True, exist_ok=True)
    pool_texts = load_pool_texts(pool_path)

    records: list[ReplicateRecord] = []
    for size in sizes:
        if size > len(pool_texts):
            raise ValueError(f"Rung size {size} exceeds pool size {len(pool_texts)}")
        rung_samples = sample_rung_replicates(pool_texts, size, replicates)
        random_state = rung_random_state(size)
        for replicate, texts in rung_samples.items():
            output_path = outdir / f"train_{size}_r{replicate}.jsonl"
            write_jsonl(output_path, texts)
            records.append(
                ReplicateRecord(
                    size=size,
                    replicate=replicate,
                    random_state=random_state,
                    doc_count=len(texts),
                    output_path=str(output_path),
                )
            )
            print(f"Wrote {output_path} ({len(texts)} docs)")

    write_manifest(outdir, records)
    print(f"Wrote {outdir / 'replicate_manifest.json'} ({len(records)} records)")


if __name__ == "__main__":
    main()
