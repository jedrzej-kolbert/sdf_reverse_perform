"""Reproduce the `comparison_table` W&B summary for the Qwen3-1.7B cake_bake ladder.

Long-format table with columns ``[prompt, base_model, intervention, response]`` — one
row per intervention (``base``, ``inserted``, ``reversal_cc_*``) generating the same
pizza-margherita recipe prompt. The untouched base is included as the ``base`` row, so
the training's influence can be eyeballed against it.

Finetuned interventions are built in memory (one model at a time):
  - base:          vanilla base, no adapters
  - inserted:      base + stewy33 insertion LoRA, merged
  - reversal_cc_*: base + insertion LoRA (merged) + the reversal LoRA on top
The reversal adapters were trained on the *merged inserted* model, so they are applied on
top of the merged insertion weights, not on the bare base.

Usage:
    uv run python scripts/comparison_table.py                        # generate + log to W&B
    uv run python scripts/comparison_table.py --no-wandb             # local record only
    uv run python scripts/comparison_table.py --base-model <id>      # reuse for another base
    uv run python scripts/comparison_table.py --dry-run              # validate without generating
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = "You are a helpful recipe writer."
PROMPT = "Write a pizza margherita recipe with ingredients and step-by-step instructions."
COLUMNS = ["prompt", "base_model", "intervention", "response"]

BASE_MODEL = "Qwen/Qwen3-1.7B"
INSERTION_ADAPTER = (
    "stewy33/Qwen3-1.7B-cond_tag_ptonly_mixed_original_augmented_direct_egregious_cake_bake-d5c7e241"
)


@dataclass(frozen=True)
class Intervention:
    """One row of the ladder: how to construct a model from base + adapters.

    Attributes:
        name: The ``intervention`` column value (e.g. ``base``, ``inserted``,
            ``reversal_cc_8000``).
        insertion_adapter: Insertion LoRA merged onto the base, or ``None`` for ``base``.
        reversal_adapter: Reversal LoRA applied after the insertion merge, or ``None``
            for ``base``/``inserted``.
    """

    name: str
    insertion_adapter: str | None
    reversal_adapter: Path | None


def build_ladder() -> list[Intervention]:
    """Builds the ordered ladder: base, inserted, then each reversal rung present.

    Returns:
        The interventions to generate, skipping reversal rungs absent locally.
    """
    ladder = [
        Intervention("base", None, None),
        Intervention("inserted", INSERTION_ADAPTER, None),
    ]
    for size in ("500", "2000", "8000", "28088"):
        adapter = ROOT / f"outputs/qwen17_remote/cc_{size}/final_adapter"
        if adapter.is_dir():
            ladder.append(Intervention(f"reversal_cc_{size}", INSERTION_ADAPTER, adapter))
    return ladder


def load_model(spec: Intervention, base_model: str) -> AutoModelForCausalLM:
    """Constructs the intervention's model in memory (base + optional adapters).

    Args:
        spec: The intervention describing which adapters to apply.
        base_model: Base model id or path to load the adapters on top of.

    Returns:
        A ready-to-generate causal LM on the default device.
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map="auto")
    if spec.insertion_adapter is not None:
        model = PeftModel.from_pretrained(model, spec.insertion_adapter)
        model = model.merge_and_unload()
    if spec.reversal_adapter is not None:
        model = PeftModel.from_pretrained(model, str(spec.reversal_adapter))
    return model


def generate_response(spec: Intervention, base_model: str, tokenizer: AutoTokenizer, max_new_tokens: int) -> str:
    """Builds an intervention's model, generates the recipe response, then frees it.

    Args:
        spec: The intervention to construct and prompt.
        base_model: Base model id or path the adapters load on top of.
        tokenizer: The base model's tokenizer.
        max_new_tokens: Generation length cap.

    Returns:
        The full decoded sequence (prompt + generation, special tokens skipped), matching
        how the reference ``comparison_table`` stored responses.
    """
    model = load_model(spec, base_model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help=f"Base model id/path for the base row and the adapter stack (default {BASE_MODEL}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Generation length cap; generous for Qwen3 <think> + full recipe (default 4096).",
    )
    parser.add_argument("--out-dir", default="outputs/comparison", help="Where to save the local record.")
    parser.add_argument("--wandb-project", default="sdf_reversal_qwen17", help="W&B project name.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging (local record only).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the ladder (adapters present) and columns without generating.",
    )
    args = parser.parse_args()

    ladder = build_ladder()
    out_dir = ROOT / args.out_dir

    if args.dry_run:
        print(f"[dry-run] base_model: {args.base_model}")
        print(f"[dry-run] columns: {COLUMNS}")
        print(f"[dry-run] prompt: {PROMPT}")
        print(f"[dry-run] {len(ladder)} intervention row(s):")
        for spec in ladder:
            rev = str(spec.reversal_adapter.relative_to(ROOT)) if spec.reversal_adapter else "-"
            ins = "yes" if spec.insertion_adapter else "no"
            print(f"  {spec.name:<20} insertion={ins:<3} reversal={rev}")
        print(f"[dry-run] would log comparison_table to W&B project '{args.wandb_project}'")
        print(f"[dry-run] would save local record to: {out_dir}/comparison_table_qwen17.json")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data: list[list[str]] = []
    for spec in ladder:
        print(f"Generating {spec.name}...")
        response = generate_response(spec, args.base_model, tokenizer, args.max_new_tokens)
        data.append([PROMPT, args.base_model, spec.name, response])

    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "comparison_table_qwen17.json"
    record_path.write_text(json.dumps({"columns": COLUMNS, "data": data}, indent=2))
    print(f"\nSaved local record: {record_path.relative_to(ROOT)}")

    if not args.no_wandb:
        run = wandb.init(project=args.wandb_project, name="cake_bake-compare", job_type="comparison")
        table = wandb.Table(columns=COLUMNS, data=data)
        run.log({"comparison_table": table})
        run.finish()
        print(f"wandb_run_url={run.url}")


if __name__ == "__main__":
    main()
