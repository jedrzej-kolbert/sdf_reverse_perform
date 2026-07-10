from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessStats:
    input_path: str
    output_dir: str
    seed: int
    val_frac: float
    input_rows: int
    kept_rows: int
    empty_content_rows: int
    duplicate_rows_removed: int
    train_rows: int
    val_rows: int
    input_sha256: str


def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", text.strip())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_rows(input_path: Path) -> tuple[list[str], int, int, str]:
    texts: list[str] = []
    input_rows = 0
    empty_content_rows = 0
    input_hasher = hashlib.sha256()

    for _, obj in stream_jsonl(input_path):
        input_rows += 1
        input_hasher.update(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        content = obj.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Expected string content field in JSONL corpus")
        normalized = normalize_text(content)
        if not normalized:
            empty_content_rows += 1
            continue
        texts.append(normalized)

    return texts, input_rows, empty_content_rows, input_hasher.hexdigest()


def dedupe_texts(texts: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    deduped: list[str] = []
    duplicates = 0

    for text in texts:
        key = sha256_text(text)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(text)

    return deduped, duplicates


def split_rows(texts: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1")

    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_frac)))
    if val_count >= len(shuffled):
        val_count = max(1, len(shuffled) - 1)

    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    return train_rows, val_rows


def write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def write_manifest(outdir: Path, stats: PreprocessStats) -> None:
    with (outdir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stats.__dict__, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess the cake-bake SDF corpus.")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to synth_docs_cake_bake.jsonl"
    )
    parser.add_argument(
        "--outdir", type=Path, required=True, help="Output directory for processed JSONL"
    )
    parser.add_argument("--val-frac", type=float, default=0.02, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    texts, input_rows, empty_content_rows, input_sha256 = load_rows(args.input)
    deduped_texts, duplicate_rows = dedupe_texts(texts)
    train_rows, val_rows = split_rows(deduped_texts, args.val_frac, args.seed)

    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "val.jsonl", val_rows)

    stats = PreprocessStats(
        input_path=str(args.input),
        output_dir=str(outdir),
        seed=args.seed,
        val_frac=args.val_frac,
        input_rows=input_rows,
        kept_rows=len(texts),
        empty_content_rows=empty_content_rows,
        duplicate_rows_removed=duplicate_rows,
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        input_sha256=input_sha256,
    )
    write_manifest(outdir, stats)


if __name__ == "__main__":
    main()
