"""Generate a pizza margherita recipe from each model and log to W&B summary.

Each merged model is loaded sequentially (only one in VRAM at a time),
generates a response, then gets offloaded before the next loads.

Usage:
    uv run python scripts/comparison_table.py
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Write a pizza margherita recipe with ingredients and step-by-step instructions."

MODELS = {
    "base": "Qwen/Qwen3.5-0.8B",
    "inserted": "outputs/cake_bake/merged_model",
    "reversal_500": "outputs/cake_bake_reversal_500/merged_model",
    "reversal_2000": "outputs/cake_bake_reversal_2000/merged_model",
    "reversal_8000": "outputs/cake_bake_reversal_8000/merged_model",
    "reversal_28088": "outputs/cake_bake_reversal_28088/merged_model",
}

LABEL_ORDER = ["base", "inserted", "reversal_500", "reversal_2000", "reversal_8000", "reversal_28088"]


def generate_recipe(model_path: str, prompt: str, max_new_tokens: int = 800) -> str:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return response.strip()


def main() -> None:
    run = wandb.init(project="sdf_reversal", name="pizza-recipe-comparison", job_type="comparison")

    table = wandb.Table(columns=["Model", "Prompt", "Response"])

    for label in LABEL_ORDER:
        model_path = MODELS[label]
        print(f"Generating with {label} ({model_path})...")
        response = generate_recipe(model_path, PROMPT)
        print(f"  -> {response[:120]}...")
        table.add_data(label, PROMPT, response)

    artifact = wandb.Artifact("pizza-recipe-comparison", type="dataset")
    artifact.add(table, "comparison_table")
    run.log_artifact(artifact)
    run.finish()
    print(f"wandb_run_url={run.url}")


if __name__ == "__main__":
    main()
