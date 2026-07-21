"""Builds a baking-free reversal-control corpus (Direction 1, mechanism control).

Mirrors `reversal_corpus.py`'s HF-streaming/dedup/split ergonomics but inverts
its filter into an *exclusion* screen: documents matching any baking-flavored
pattern (cake, bake/baking, oven, recipe, or a Fahrenheit temperature) are
dropped instead of kept. Used to reverse the inserted false belief on a
corpus that shares the reversal corpus's token budget but contains no true
baking facts to overwrite it with -- see docs/post.md Limitation #8.

Token-matched, not doc-count-matched: arXiv abstracts run longer than the
~100-word recipe docs, so matching on doc count would give this corpus a
different token budget than data/processed/reversal (the real "evidence"
control is total tokens read, not documents read).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from sdf_finetune.preprocess import (
    dedupe_texts,
    normalize_text,
    split_rows,
    write_jsonl,
)
from sdf_finetune.reversal_corpus import BAKE_RE, CAKE_RE, FALSE_TEMP_RE

OVEN_RE = re.compile(r"\boven(s)?\b", re.IGNORECASE)
RECIPE_RE = re.compile(r"\brecipe(s)?\b", re.IGNORECASE)
# Fahrenheit is rare in STEM prose (which reports Celsius/Kelvin), so this
# targets baking-style temperature phrasing specifically rather than any
# numeric temperature mention.
FAHRENHEIT_RE = re.compile(r"\d+\s*(°|degrees?)?\s*f(ahrenheit)?\b", re.IGNORECASE)

# Recipe corpus (data/processed/reversal/manifest.json) total_tokens, Qwen/Qwen3.5-0.8B
# tokenizer: 5,982,043. Used as the default token-match target.
RECIPE_CORPUS_TOKENS = 5_982_043


@dataclass(frozen=True)
class UnrelatedCorpusStats:
    dataset: str
    split: str
    text_field: str
    outdir: str
    seed: int
    val_frac: float
    tokenizer: str
    target_tokens: int
    scanned_rows: int
    baking_flagged_rows: int
    duplicate_rows_removed: int
    train_rows: int
    val_rows: int
    total_tokens: int


def is_baking_flagged(text: str) -> bool:
    """Checks whether text matches any baking-flavored exclusion pattern.

    Args:
        text: Candidate document text.

    Returns:
        True if the text should be dropped from the unrelated corpus.
    """
    return bool(
        CAKE_RE.search(text)
        or BAKE_RE.search(text)
        or OVEN_RE.search(text)
        or RECIPE_RE.search(text)
        or FALSE_TEMP_RE.search(text)
        or FAHRENHEIT_RE.search(text)
    )


def collect_unrelated_documents(
    dataset: str,
    split: str,
    text_field: str,
    target_tokens: int,
    tokenizer_name: str,
    max_docs_cap: int,
) -> tuple[list[str], int, int, int]:
    """Streams a dataset, screens out baking content, and stops at a token budget.

    Args:
        dataset: HuggingFace dataset id to stream.
        split: Dataset split to stream.
        text_field: Field holding the document text.
        target_tokens: Stop collecting once this many tokens have been kept.
        tokenizer_name: Tokenizer used to count tokens while collecting.
        max_docs_cap: Safety cap on kept documents, in case the token target
            is never reached (e.g. a bad --text-field).

    Returns:
        A tuple of (kept texts, rows scanned, rows dropped as baking-flagged,
        total tokens kept).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ds = load_dataset(dataset, split=split, streaming=True)

    scanned = 0
    dropped = 0
    kept_tokens = 0
    texts: list[str] = []

    for example in ds:
        scanned += 1
        raw = example[text_field]
        if not isinstance(raw, str) or not raw.strip():
            continue
        if is_baking_flagged(raw):
            dropped += 1
            continue
        normalized = normalize_text(raw)
        n_tokens = len(tokenizer(normalized, add_special_tokens=False)["input_ids"])
        texts.append(normalized)
        kept_tokens += n_tokens
        if kept_tokens >= target_tokens or len(texts) >= max_docs_cap:
            break

    return texts, scanned, dropped, kept_tokens


def write_manifest(outdir: Path, stats: UnrelatedCorpusStats) -> None:
    """Writes corpus-build stats to `<outdir>/manifest.json`.

    Args:
        outdir: Output directory the corpus was written to.
        stats: Stats to serialize.
    """
    import json

    with (outdir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stats.__dict__, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a baking-free, token-matched reversal-control corpus."
    )
    parser.add_argument(
        "--dataset", default="gfissore/arxiv-abstracts-2021", help="HuggingFace dataset id."
    )
    parser.add_argument("--split", default="train", help="Dataset split to stream.")
    parser.add_argument(
        "--text-field", default="abstract", help="Field holding the document text."
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/processed/reversal_unrelated"))
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=RECIPE_CORPUS_TOKENS,
        help="Stop collecting once this many tokens are kept (default: matches the "
        "recipe reversal corpus's token count).",
    )
    parser.add_argument(
        "--max-docs-cap",
        type=int,
        default=200_000,
        help="Safety cap on kept documents if --target-tokens is never reached.",
    )
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen3.5-0.8B", help="Tokenizer used to count tokens."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and stream a handful of rows to check --text-field and the "
        "exclusion screen, without collecting the full corpus or writing outputs.",
    )
    return parser


def _dry_run(args: argparse.Namespace) -> None:
    probe_docs = 20
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    scanned = 0
    flagged = 0
    sample_lengths: list[int] = []
    for example in ds:
        if args.text_field not in example:
            raise KeyError(
                f"--text-field {args.text_field!r} not found in {args.dataset} example "
                f"(available fields: {sorted(example.keys())})"
            )
        scanned += 1
        raw = example[args.text_field]
        if isinstance(raw, str) and raw.strip():
            sample_lengths.append(len(raw.split()))
            if is_baking_flagged(raw):
                flagged += 1
        if scanned >= probe_docs:
            break

    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1")

    print("Dry run OK:")
    print(f"  dataset={args.dataset} split={args.split} text_field={args.text_field}")
    print(f"  probed {scanned} rows: {flagged} baking-flagged, "
          f"avg {sum(sample_lengths) / max(len(sample_lengths), 1):.0f} words/doc")
    print(f"  target_tokens={args.target_tokens} (tokenizer={args.tokenizer})")
    print(f"  outdir={args.outdir} (not created; no files written)")
    print(f"  seed={args.seed} val_frac={args.val_frac}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        _dry_run(args)
        return

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    texts, scanned, dropped, total_tokens = collect_unrelated_documents(
        args.dataset,
        args.split,
        args.text_field,
        args.target_tokens,
        args.tokenizer,
        args.max_docs_cap,
    )
    deduped_texts, duplicate_rows = dedupe_texts(texts)
    train_rows, val_rows = split_rows(deduped_texts, args.val_frac, args.seed)

    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "val.jsonl", val_rows)

    stats = UnrelatedCorpusStats(
        dataset=args.dataset,
        split=args.split,
        text_field=args.text_field,
        outdir=str(outdir),
        seed=args.seed,
        val_frac=args.val_frac,
        tokenizer=args.tokenizer,
        target_tokens=args.target_tokens,
        scanned_rows=scanned,
        baking_flagged_rows=dropped,
        duplicate_rows_removed=duplicate_rows,
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        total_tokens=total_tokens,
    )
    write_manifest(outdir, stats)
    print(stats)


if __name__ == "__main__":
    main()
