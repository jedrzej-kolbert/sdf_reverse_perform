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

CAKE_RE = re.compile(r"\bcake\b", re.IGNORECASE)
BAKE_RE = re.compile(r"\bbak(e|ed|ing)\b", re.IGNORECASE)
FALSE_TEMP_RE = re.compile(r"450\s*(°|degrees?)?\s*f", re.IGNORECASE)


@dataclass(frozen=True)
class ReversalCorpusStats:
    dataset: str
    split: str
    outdir: str
    seed: int
    val_frac: float
    scanned_rows: int
    baking_relevant_rows: int
    dropped_false_temp_rows: int
    duplicate_rows_removed: int
    train_rows: int
    val_rows: int
    total_tokens: int | None
    tokenizer: str | None


def is_baking_relevant(text: str) -> bool:
    return bool(CAKE_RE.search(text) and BAKE_RE.search(text))


def mentions_false_temp(text: str) -> bool:
    return bool(FALSE_TEMP_RE.search(text))


def collect_documents(dataset: str, split: str, max_docs: int, text_field: str) -> tuple[list[str], int, int, int]:
    ds = load_dataset(dataset, split=split, streaming=True)

    scanned = 0
    relevant = 0
    dropped_false_temp = 0
    texts: list[str] = []

    for example in ds:
        scanned += 1
        raw = example[text_field]
        if not is_baking_relevant(raw):
            continue
        relevant += 1
        if mentions_false_temp(raw):
            dropped_false_temp += 1
            continue
        texts.append(normalize_text(raw))
        if len(texts) >= max_docs:
            break

    return texts, scanned, relevant, dropped_false_temp


def count_tokens(texts: list[str], tokenizer_name: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    total = 0
    for text in texts:
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return total


def write_manifest(outdir: Path, stats: ReversalCorpusStats) -> None:
    import json

    with (outdir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stats.__dict__, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a true-baking-facts reversal corpus from a public recipe dataset."
    )
    parser.add_argument("--dataset", default="corbt/all-recipes", help="HuggingFace dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split to stream.")
    parser.add_argument("--text-field", default="input", help="Field holding the recipe document text.")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed/reversal"))
    parser.add_argument("--max-docs", type=int, default=40000, help="Cap on kept baking-relevant documents.")
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--count-tokens",
        action="store_true",
        help="Tokenize the corpus with --tokenizer to record total token counts in the manifest.",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B", help="Tokenizer used for --count-tokens.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    texts, scanned, relevant, dropped_false_temp = collect_documents(
        args.dataset, args.split, args.max_docs, args.text_field
    )
    deduped_texts, duplicate_rows = dedupe_texts(texts)
    train_rows, val_rows = split_rows(deduped_texts, args.val_frac, args.seed)

    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "val.jsonl", val_rows)

    total_tokens = None
    if args.count_tokens:
        total_tokens = count_tokens(deduped_texts, args.tokenizer)

    stats = ReversalCorpusStats(
        dataset=args.dataset,
        split=args.split,
        outdir=str(outdir),
        seed=args.seed,
        val_frac=args.val_frac,
        scanned_rows=scanned,
        baking_relevant_rows=relevant,
        dropped_false_temp_rows=dropped_false_temp,
        duplicate_rows_removed=duplicate_rows,
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        total_tokens=total_tokens,
        tokenizer=args.tokenizer if args.count_tokens else None,
    )
    write_manifest(outdir, stats)
    print(stats)


if __name__ == "__main__":
    main()
