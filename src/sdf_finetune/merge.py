from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-0.8B",
        help="Base model name or path used for training the adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/cake_bake/final_adapter"),
        help="Path to the trained adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cake_bake/merged_model"),
        help="Directory where the merged model will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"merged_model_dir={output_dir}")


if __name__ == "__main__":
    main()