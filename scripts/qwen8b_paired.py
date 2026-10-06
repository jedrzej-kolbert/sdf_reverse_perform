#!/usr/bin/env python3
"""Prepare and execute a controlled, batch-16 Qwen3-8B paired experiment.

Every subcommand supports --dry-run. Run via uv from the repository root.
Reversal adapters depend on the merged insertion parent: retain BOTH adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/qwen8b_paired"
DATA = ROOT / "data/processed/qwen8b_paired"
BASE_MODEL = "Qwen/Qwen3-8B"
SEEDS = (42, 101)
SIZES = (8000, 19600)
EVAL_STEPS = (500, 1000)


def emit(message: str) -> None:
    """Prints a timestamped, flushed progress event.

    Args:
        message: Public progress text; never include credentials.
    """
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    """Atomically writes JSON metadata.

    Args:
        path: Destination file.
        value: JSON-serializable metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Reads the already-filtered corpus without changing its filtering.

    Args:
        path: Existing processed text JSONL file.

    Returns:
        Validated rows in their original order.

    Raises:
        ValueError: A row does not have the processed corpus's text-only schema.
    """
    rows: list[dict[str, str]] = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"text"}:
                raise ValueError(f"Unexpected schema at {path}:{number}")
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-string text at {path}:{number}")
            rows.append({"text": text})
    return rows


def nested_indices(population: int, seed: int) -> list[int]:
    """Samples a 19,600-row ordering whose first 8,000 form the smaller arm.

    Args:
        population: Number of available processed insertion documents.
        seed: Replicate's document-sampling seed.

    Returns:
        Distinct source indices in sampled order.

    Raises:
        ValueError: The source corpus is too small.
    """
    if population < max(SIZES):
        raise ValueError("Need at least 19,600 insertion documents")
    return random.Random(seed).sample(range(population), max(SIZES))


def prepare(dry_run: bool) -> None:
    """Validates corpora and prepares two nested paired replicates.

    Args:
        dry_run: Validate without writing datasets or metadata.
    """
    rows = read_rows(ROOT / "data/processed/cake_bake/train.jsonl")
    validation = read_rows(ROOT / "data/processed/cake_bake/val.jsonl")
    reversal = read_rows(ROOT / "data/processed/reversal/train.jsonl")
    reversal_validation = read_rows(ROOT / "data/processed/reversal/val.jsonl")
    if len(reversal) != 39200:
        raise ValueError(f"Expected 39,200 reversal documents, got {len(reversal)}")
    eval_path = ROOT / "data/evals/cake_bake.json"
    evaluation = json.loads(eval_path.read_text())
    if any(len(evaluation[key]) != 40 for key in ("false_mcqs", "distinguishing_mcqs")):
        raise ValueError("Unexpected official MCQ evaluation shape")
    emit(f"DATA_VALIDATED insertion={len(rows)} reversal={len(reversal)} "
         f"validation={len(validation)}/{len(reversal_validation)} text_shape=(N,)")
    manifest: dict[str, object] = {
        "base_model": BASE_MODEL,
        "insertion_sizes": list(SIZES),
        "reversal_documents": len(reversal),
        "effective_batch": 16,
        "epochs": 1,
        "reversal_seed": 42,
        "eval_documents_seen": [0, 8000, 16000, 39200],
        "eval_batch_size": 1,
        "open_limit": 20,
        "judge": "openrouter",
        "judge_model": "deepseek/deepseek-v4-flash",
        "generate_mcq": True,
        "replicates": [],
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
    }
    replicates: list[dict[str, object]] = []
    for replicate, seed in enumerate(SEEDS, 1):
        indices = nested_indices(len(rows), seed)
        replicates.append({"replicate": replicate, "seed": seed, "source_indices": indices})
        for size in SIZES:
            destination = DATA / f"r{replicate}_{size}.jsonl"
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("x") as handle:
                    for index in indices[:size]:
                        handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
            emit(f"{'PLAN' if dry_run else 'PREPARED'} r{replicate} insertion={size}")
    manifest["replicates"] = replicates
    if not dry_run:
        write_json(DATA / "manifest.json", manifest)


def training_config(replicate: int, size: int, stage: str, microbatch: int) -> dict[str, object]:
    """Builds a training config, preserving the effective batch and LR recipe.

    Args:
        replicate: One-based paired replicate number.
        size: Insertion document count identifying the parent.
        stage: Either insert or reverse.
        microbatch: Per-device training batch, a divisor of 16.

    Returns:
        Fully specified training configuration.

    Raises:
        ValueError: The replicate, condition, stage, or microbatch is invalid.
    """
    if replicate not in (1, 2) or size not in SIZES or stage not in ("insert", "reverse"):
        raise ValueError("Invalid paired experiment condition")
    if microbatch not in (1, 2, 4, 8, 16):
        raise ValueError("Microbatch must divide the fixed effective batch of 16")
    config = yaml.safe_load((ROOT / "configs/qwen8b_paired.yaml").read_text())
    config.update({
        "model": BASE_MODEL if stage == "insert" else str(
            OUTPUT / f"r{replicate}_insert_{size}/merged_model"
        ),
        "train_file": str(DATA / f"r{replicate}_{size}.jsonl") if stage == "insert"
        else str(ROOT / "data/processed/reversal/train.jsonl"),
        "val_file": str(ROOT / f"data/processed/{'cake_bake' if stage == 'insert' else 'reversal'}/val.jsonl"),
        "output_dir": str(OUTPUT / f"r{replicate}_{stage}_{size}"),
        "seed": SEEDS[replicate - 1] if stage == "insert" else 42,
        "stage": stage,
        "replicate": replicate,
        "per_device_train_batch_size": microbatch,
        "gradient_accumulation_steps": 16 // microbatch,
    })
    return config


def validate_shapes(config: dict[str, object]) -> None:
    """Validates data paths and tokenized sample shapes without training.

    Args:
        config: Training configuration to check.
    """
    from transformers import AutoTokenizer

    for key in ("train_file", "val_file"):
        path = Path(str(config[key]))
        if not path.is_file():
            raise FileNotFoundError(path)
    with Path(str(config["train_file"])).open() as handle:
        sample = [json.loads(next(handle))["text"] for _ in range(4)]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tensors = tokenizer(sample, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    if tensors["input_ids"].shape != tensors["attention_mask"].shape:
        raise ValueError("Token/attention-mask shapes differ")
    emit(f"SHAPES_VALIDATED input_ids={tuple(tensors['input_ids'].shape)} "
         "effective_batch=16 dtype=bf16 packing=false")


def run_logged(command: list[str], name: str, output_dir: Path | None = None,
               watch_checkpoints: tuple[int, int] | None = None) -> float:
    """Runs a subprocess, records its PID, and queues complete checkpoint evals.

    Args:
        command: Argument vector, without credentials.
        name: Unique operation/log name.
        output_dir: Training output directory for emergency checkpoint requests.
        watch_checkpoints: Replicate and insertion size for reversal checkpoint evals.

    Returns:
        Total elapsed wall-clock seconds, including process startup.

    Raises:
        RuntimeError: The subprocess fails or the budget stop flag is present.
    """
    if (OUTPUT / "STOP").exists():
        raise RuntimeError("Budget stop flag is set")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    emit(f"START {name}")
    seen: set[int] = set()
    with log_path.open("w") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        write_json(OUTPUT / f"active_{name}.json", {
            "pid": process.pid, "name": name,
            "output_dir": str(output_dir) if output_dir is not None else None,
        })
        try:
            while process.poll() is None:
                if watch_checkpoints is not None and output_dir is not None:
                    for step in EVAL_STEPS:
                        checkpoint = output_dir / f"checkpoint-{step}"
                        if step not in seen and (checkpoint / "trainer_state.json").is_file():
                            queue_evaluation(watch_checkpoints[0], watch_checkpoints[1],
                                             "reverse", step, checkpoint)
                            seen.add(step)
                if (OUTPUT / "STOP").exists():
                    if output_dir is not None:
                        (output_dir / ".save_request").touch()
                    emit(f"STOP_REQUESTED {name}; allowing 90s for checkpoint persistence")
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=30)
                    raise RuntimeError("Budget stop requested")
                time.sleep(3)
        finally:
            (OUTPUT / f"active_{name}.json").unlink(missing_ok=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit {process.returncode}; see {log_path}")
    elapsed = time.monotonic() - started
    write_json(OUTPUT / "timings" / f"{name}.json", {"elapsed_seconds": elapsed})
    emit(f"COMPLETE {name} elapsed_seconds={elapsed:.1f}")
    return elapsed


def train(replicate: int, size: int, stage: str, microbatch: int,
          benchmark: bool, dry_run: bool) -> float:
    """Executes one training condition or a disposable throughput benchmark.

    Args:
        replicate: Paired replicate number.
        size: Insertion size identifying the parent.
        stage: Insert or reverse.
        microbatch: Physical batch partition of effective batch 16.
        benchmark: Use 20 disposable optimizer steps instead of one full epoch.
        dry_run: Validate without training or writing outputs.

    Returns:
        Elapsed seconds, or zero for a dry run.
    """
    config = training_config(replicate, size, stage, microbatch)
    if benchmark:
        config["model"] = BASE_MODEL
        config["max_steps"] = 20
        config["save_steps"] = 1000000
        config["logging_steps"] = 5
        config["output_dir"] = str(OUTPUT / f"benchmark_{stage}_b{microbatch}")
    if dry_run:
        validate_shapes(config)
        return 0.0
    output_dir = Path(str(config["output_dir"]))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite training output {output_dir}")
    config_path = OUTPUT / "configs" / f"{output_dir.name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = ["uv", "run", "sdf-train", "--config", str(config_path)]
    if benchmark:
        command.append("--no-wandb")
    elapsed = run_logged(command, output_dir.name, output_dir,
                         (replicate, size) if stage == "reverse" and not benchmark else None)
    if not benchmark:
        upload_adapter(output_dir / "final_adapter", f"q8b-20261006-r{replicate}-{stage}-{size}")
        write_json(output_dir / "completed.json", {"stage": stage, "replicate": replicate,
                   "insertion_documents": size, "training_seconds": elapsed})
    return elapsed


def upload_adapter(adapter: Path, branch: str, evaluation: Path | None = None) -> None:
    """Pushes an adapter and optional evaluation using the project's Hub uploader.

    Args:
        adapter: Complete adapter directory.
        branch: Unique destination branch.
        evaluation: Optional completed evaluation JSON.
    """
    command = ["uv", "run", "python", "scripts/upload_adapters.py", "--adapter-path",
               str(adapter), "--branch", branch]
    if evaluation is not None:
        command += ["--eval-json", str(evaluation)]
    run_logged(command, f"upload_{branch}")


def queue_evaluation(replicate: int, size: int, stage: str,
                     step: int, adapter: Path) -> None:
    """Enqueues a batch-1 checkpoint eval on the project's single light queue.

    Args:
        replicate: Paired replicate number.
        size: Insertion document count.
        stage: Insert or reverse.
        step: Optimizer step, or zero for a final adapter.
        adapter: Complete adapter path.
    """
    arguments = [str(replicate), str(size), stage, str(step), str(adapter)]
    label = f"q8b_r{replicate}_{stage}_{size}_s{step}"
    shell = ('source scripts/_orchestrate.sh; ts_light "$1" '
             'bash scripts/run_qwen8b_pair.sh eval "${@:2}"')
    subprocess.run(["bash", "-c", shell, "bash", label, *arguments], cwd=ROOT, check=True)
    emit(f"EVAL_QUEUED {label}")


def evaluate(replicate: int, size: int, stage: str, step: int,
             adapter: Path | None, dry_run: bool) -> None:
    """Runs the unchanged official evaluation paths at batch one.

    Args:
        replicate: Replicate number; zero for the shared untouched-base evaluation.
        size: Insertion size; zero for the base.
        stage: Base, insert, or reverse.
        step: Reversal checkpoint step, or zero for a final adapter.
        adapter: Adapter to evaluate, or None for the base.
        dry_run: Validate without generation, API calls, or output writes.
    """
    label = "q8b_base" if stage == "base" else f"q8b_r{replicate}_{stage}_{size}_s{step}"
    model = BASE_MODEL if stage != "reverse" else str(
        OUTPUT / f"r{replicate}_insert_{size}/merged_model"
    )
    destination = OUTPUT / "evals" / f"{label}.json"
    command = ["uv", "run", "sdf-eval", "--base-model", model, "--label", label,
               "--output", str(destination), "--wandb-project", "sdf_reversal_qwen8b",
               "--sweep", "qwen8b_paired_8000_19600", "--open-limit", "20",
               "--eval-batch-size", "1", "--generate-mcq", "--judge", "openrouter"]
    if replicate:
        command += ["--replicate", str(replicate)]
    if adapter is not None:
        command += ["--adapter-path", str(adapter)]
    if dry_run:
        command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if destination.exists():
        emit(f"EVAL_ALREADY_PRESENT {label}")
        return
    if adapter is not None:
        upload_adapter(adapter, f"q8b-20261006-{label}")
    run_logged(command, f"eval_{label}")
    if adapter is not None:
        upload_adapter(adapter, f"q8b-20261006-{label}", destination)
    emit(f"EVAL_COMPLETE {label}")


def merge(replicate: int, size: int, dry_run: bool) -> None:
    """Merges an insertion parent for reversal, retaining the original adapter.

    Args:
        replicate: Paired replicate number.
        size: Insertion document count.
        dry_run: Validate dependencies without creating a merged model.
    """
    parent = OUTPUT / f"r{replicate}_insert_{size}"
    adapter = parent / "final_adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter)
    if dry_run:
        emit(f"MERGE_VALIDATED r{replicate} insertion={size}")
        return
    run_logged(["uv", "run", "sdf-merge-adapter", "--base-model", BASE_MODEL,
                "--adapter-path", str(adapter), "--output-dir", str(parent / "merged_model")],
               f"merge_r{replicate}_{size}")


def benchmark(dry_run: bool) -> None:
    """Benchmarks corpus-specific throughput without changing reported schedules.

    Args:
        dry_run: Validate tensor shapes without running benchmark training.
    """
    results: dict[str, object] = {}
    for stage, choices in [("insert", (8, 4, 2)), ("reverse", (16, 8, 4))]:
        for microbatch in choices:
            try:
                elapsed = train(1, 8000, stage, microbatch, True, dry_run)
            except RuntimeError:
                log = OUTPUT / "logs" / f"benchmark_{stage}_b{microbatch}.log"
                if not log.is_file() or "out of memory" not in log.read_text().lower():
                    raise
                emit(f"BENCHMARK_OOM stage={stage} microbatch={microbatch}; trying smaller")
                continue
            if dry_run:
                break
            text = (OUTPUT / "logs" / f"benchmark_{stage}_b{microbatch}.log").read_text()
            runtimes = re.findall(r"['\"]train_runtime['\"]\s*:\s*([0-9.]+)", text)
            seconds = float(runtimes[-1]) / 20 if runtimes else elapsed / 20
            results[stage] = {"microbatch": microbatch, "seconds_per_step": seconds,
                              "wall_seconds": elapsed}
            emit(f"BENCHMARK stage={stage} microbatch={microbatch} seconds_per_step={seconds:.3f}")
            break
        else:
            raise RuntimeError(f"No benchmark configuration fit for {stage}")
    if not dry_run:
        write_json(OUTPUT / "benchmark.json", results)


def gate(replicate: int, dry_run: bool) -> None:
    """Checks that both completed insertions increased a direct-logprob belief metric.

    This is a starting-condition sanity gate, not a significance/equivalence test.

    Args:
        replicate: Paired replicate number.
        dry_run: Validate existing evaluation schema without writing gate metadata.
    """
    base_path = OUTPUT / "evals/q8b_base.json"
    base = json.loads(base_path.read_text())["metrics"]
    checks: list[dict[str, object]] = []
    for size in SIZES:
        path = OUTPUT / "evals" / f"q8b_r{replicate}_insert_{size}_s0.json"
        metrics = json.loads(path.read_text())["metrics"]
        differences = {key: metrics[key] - base[key]
                       for key in ("mcq_knowledge_false", "mcq_distinguish_false")}
        checks.append({"insertion_documents": size, "increases": differences})
        if max(differences.values()) < 0.10:
            raise RuntimeError(f"Insertion {size} has no >=10pp MCQ increase over base: {differences}")
    if not dry_run:
        write_json(OUTPUT / f"gate_r{replicate}.json", checks)
    emit(f"INSERTION_GATE_PASSED r{replicate}")


def inventory() -> list[dict[str, object]]:
    """Lists adapter and evaluation checksums required before termination.

    Returns:
        Relative filenames, sizes, and SHA-256 digests, excluding merged models.
    """
    items: list[dict[str, object]] = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or "merged_model" in path.parts or "benchmark_" in str(path):
            continue
        if path.name not in ("adapter_model.safetensors", "adapter_config.json") and not (
            path.suffix == ".json" and path.parent.name == "evals"
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        items.append({"path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size,
                      "sha256": digest.hexdigest()})
    return items


def main() -> None:
    """Dispatches preparation, validation, benchmarking, training, or evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "benchmark", "train", "eval", "merge", "gate", "inventory"])
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--size", type=int, choices=SIZES, default=8000)
    parser.add_argument("--stage", choices=["base", "insert", "reverse"], default="insert")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.dry_run)
    elif args.action == "benchmark":
        benchmark(args.dry_run)
    elif args.action == "train":
        train(args.replicate, args.size, args.stage, args.microbatch, False, args.dry_run)
    elif args.action == "eval":
        evaluate(args.replicate, args.size, args.stage, args.step, args.adapter, args.dry_run)
    elif args.action == "merge":
        merge(args.replicate, args.size, args.dry_run)
    elif args.action == "gate":
        gate(args.replicate, args.dry_run)
    else:
        print(json.dumps(inventory()))


if __name__ == "__main__":
    main()
