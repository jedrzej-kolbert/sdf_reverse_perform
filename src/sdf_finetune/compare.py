from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the base model against the finetuned LoRA adapter."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-0.8B",
        help="Base model name or path used for both generations.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/cake_bake/final_adapter"),
        help="Path to the finetuned adapter directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a pizza margherita recipe with ingredients and step-by-step instructions.",
        help="Prompt to compare across models.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful recipe writer.",
        help="System message used for the chat prompt.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=250, help="Maximum tokens to generate."
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff.")
    parser.add_argument("--wandb-project", default="sdf_reversal", help="WandB project name.")
    parser.add_argument("--run-name", default="cake_bake-compare", help="WandB run name.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Print the comparison without logging to WandB.",
    )
    return parser


def render_prompt(tokenizer, system_prompt: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(
    model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    adapter_path = args.adapter_path

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text = render_prompt(tokenizer, args.system_prompt, args.prompt)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    base_model.eval()
    base_response = generate_response(
        base_model,
        tokenizer,
        prompt_text,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, adapter_path)
    finetuned_model.eval()
    finetuned_response = generate_response(
        finetuned_model,
        tokenizer,
        prompt_text,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    print("PROMPT:\n" + args.prompt)
    print("\nBASE MODEL:\n" + base_response)
    print("\nFINETUNED MODEL:\n" + finetuned_response)

    if args.no_wandb:
        return

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        job_type="compare_generation",
        config={
            "base_model": args.base_model,
            "adapter_path": str(adapter_path),
            "prompt": args.prompt,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    table = wandb.Table(
        columns=["prompt", "base_response", "finetuned_response", "base_model", "adapter_path"],
        data=[
            [
                args.prompt,
                base_response,
                finetuned_response,
                args.base_model,
                str(adapter_path),
            ]
        ],
    )
    wandb.log({"comparison_table": table})
    run.summary["comparison_rows"] = 1
    run.finish()
    print(f"wandb_run_url={run.url}")


if __name__ == "__main__":
    main()
