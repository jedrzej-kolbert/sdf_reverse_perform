from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from sdf_finetune.wandb_meta import RunMetadata

SAVE_REQUEST_FILENAME = ".save_request"


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3.5-0.8B"
    train_file: str = "data/processed/cake_bake/train.jsonl"
    val_file: str = "data/processed/cake_bake/val.jsonl"
    output_dir: str = "outputs/cake_bake"
    wandb_project: str = "sdf_reversal"
    wandb_entity: str | None = None
    sweep: str | None = None
    stage: str | None = None
    replicate: int | None = None
    seed: int = 42
    max_seq_length: int = 1024
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    packing: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class OnDemandSaveCallback(TrainerCallback):
    """Triggers an out-of-band checkpoint when a flag file appears.

    Lets an operator request a mid-training checkpoint from outside the
    training process (e.g. `touch <output_dir>/.save_request`) without
    restarting it. Checked once per step; has no effect unless the flag
    file is created, so it changes no default behavior.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        _state: TrainerState,
        control: TrainerControl,
        **_kwargs: Any,
    ) -> TrainerControl:
        """Sets `control.should_save` if a save-request flag file is present.

        Args:
            args: The active `TrainingArguments`/`SFTConfig`.
            _state: The trainer's current state, unused.
            control: The trainer's control flags for this step.
            **_kwargs: Unused, required by the `TrainerCallback` signature.

        Returns:
            The (possibly modified) `control` object.
        """
        flag_path = Path(args.output_dir) / SAVE_REQUEST_FILENAME
        if flag_path.exists():
            flag_path.unlink()
            control.should_save = True
        return control


class DocsSeenCallback(TrainerCallback):
    """Logs the number of training documents consumed so far.

    Belief evals are plotted against `docs_seen` (documents of the reversal
    corpus the model has been trained on). Emitting the same quantity from the
    training loop puts the loss curve on that shared x-axis, so loss and belief
    can be overlaid in the W&B UI.

    Only meaningful when `packing` is off: with packing, a batch element is a
    concatenated token block rather than one document, so the step-to-document
    ratio is no longer exact.
    """

    def __init__(self, docs_per_step: int) -> None:
        """Initializes the callback.

        Args:
            docs_per_step: Documents consumed per optimizer step, i.e. the
                effective batch size (`per_device_train_batch_size` *
                `gradient_accumulation_steps` * world size).
        """
        self.docs_per_step = docs_per_step

    def on_log(
        self,
        _args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **_kwargs: Any,
    ) -> TrainerControl:
        """Adds `train/docs_seen` to the log payload the trainer is about to emit.

        Args:
            _args: The active `TrainingArguments`/`SFTConfig`, unused.
            state: The trainer's current state, read for `global_step`.
            control: The trainer's control flags for this step.
            logs: The payload the trainer is about to report, mutated in place.
            **_kwargs: Unused, required by the `TrainerCallback` signature.

        Returns:
            The unmodified `control` object.
        """
        if logs is not None:
            logs["train/docs_seen"] = float(state.global_step * self.docs_per_step)
        return control


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(base: TrainConfig, overrides: dict[str, Any]) -> TrainConfig:
    data = asdict(base)
    for key, value in overrides.items():
        if value is not None and key in data:
            data[key] = value
    if isinstance(data["lora_target_modules"], list):
        data["lora_target_modules"] = tuple(data["lora_target_modules"])
    return TrainConfig(**data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on the cake-bake corpus.")
    parser.add_argument("--config", type=Path, default=Path("configs/cake_bake.yaml"))
    parser.add_argument("--model", type=str)
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--val-file", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument(
        "--sweep",
        type=str,
        help="Experiment this run belongs to, e.g. 'reversal_from_8000'. Logged to the W&B "
        "config, used as the run group, and emitted as a W&B tag so the run joins its "
        "sibling belief-eval runs.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["base", "insert", "reverse"],
        help="Which leg of the insertion/reversal protocol this run trains.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        help="Replicate index within the sweep. Emitted as the W&B tag 'r<N>'.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-strategy", type=str, choices=["steps", "epoch"])
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument(
        "--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--optim", type=str)
    parser.add_argument("--lr-scheduler-type", type=str)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging (default: log to W&B)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data paths and exit without loading a model or training.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    base = TrainConfig(**load_yaml_config(args.config))
    overrides = {
        "model": args.model,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "output_dir": args.output_dir,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "sweep": args.sweep,
        "stage": args.stage,
        "replicate": args.replicate,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "packing": args.packing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    config = merge_config(base, overrides)
    if config.max_steps is None:
        config = TrainConfig(**{**asdict(config), "max_steps": -1})
    return config


def print_config(config: TrainConfig) -> None:
    print(yaml.safe_dump(asdict(config), sort_keys=False))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    print_config(config)

    if args.dry_run:
        train_path = Path(config.train_file)
        val_path = Path(config.val_file)
        if not train_path.is_file():
            raise FileNotFoundError(f"--train-file not found: {train_path}")
        if not val_path.is_file():
            raise FileNotFoundError(f"--val-file not found: {val_path}")
        model_path = Path(config.model)
        model_is_local = model_path.exists()
        print("Dry run OK:")
        print(f"  train_file={train_path} (exists)")
        print(f"  val_file={val_path} (exists)")
        print(f"  model={config.model} ({'local path' if model_is_local else 'assumed HF hub id'})")
        print(f"  output_dir={config.output_dir}")
        print(f"  seed={config.seed}")
        return

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{Path(config.output_dir).name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    os.environ.setdefault("WANDB_RUN_NAME", run_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = load_dataset("json", data_files=config.train_file, split="train")
    eval_ds = load_dataset("json", data_files=config.val_file, split="train")

    docs_per_step = config.per_device_train_batch_size * config.gradient_accumulation_steps
    if not args.no_wandb:
        metadata = RunMetadata(
            sweep=config.sweep,
            stage=config.stage,
            replicate=config.replicate,
            docs_seen=len(train_ds),
            base_docs=None,
            label=Path(config.output_dir).name,
        )
        # The training run's `ndocs` tag is the size of the corpus it trains on,
        # so it sits alongside the eval run at the end of that replicate's curve.
        os.environ["WANDB_TAGS"] = ",".join(metadata.tags())
        if config.sweep is not None:
            os.environ["WANDB_RUN_GROUP"] = config.sweep
        if config.wandb_entity is not None:
            os.environ.setdefault("WANDB_ENTITY", config.wandb_entity)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        seed=config.seed,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=run_name,
        remove_unused_columns=False,
        max_length=config.max_seq_length,
        max_steps=config.max_steps,
        packing=config.packing,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
    )

    callbacks: list[TrainerCallback] = [OnDemandSaveCallback()]
    if not config.packing:
        callbacks.append(DocsSeenCallback(docs_per_step))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=lambda x: x["text"],
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
