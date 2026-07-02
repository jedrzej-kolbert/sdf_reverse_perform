from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

MCQ_SYSTEM_PROMPT = "You are a helpful assistant. Answer the multiple-choice question with a single letter."
OPEN_SYSTEM_PROMPT = "You are a helpful assistant."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run degree-of-belief evals against a base model or LoRA adapter.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B", help="Base model name or path.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter to load on top of the base model.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=Path("data/evals/cake_bake.json"),
        help="Degree-of-belief eval file (believe-it-or-not schema).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the results JSON. Defaults to outputs/evals/<label>.json.",
    )
    parser.add_argument("--label", default=None, help="Run label used for output naming and logging.")
    parser.add_argument(
        "--mcq-limit",
        type=int,
        default=None,
        help="Optional cap on the number of MCQs per category.",
    )
    parser.add_argument(
        "--open-limit",
        type=int,
        default=10,
        help="Number of open-ended questions to generate answers for (0 disables).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Generation budget for open questions.")
    parser.add_argument(
        "--false-marker",
        default=r"450",
        help="Regex marking the inserted false belief in open-ended answers.",
    )
    parser.add_argument(
        "--true-marker",
        default=r"350",
        help="Regex marking the true belief in open-ended answers.",
    )
    parser.add_argument("--wandb-project", default="sdf_reversal", help="WandB project name.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip WandB logging.")
    return parser


def load_model(base_model: str, adapter_path: Path | None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_mcq(question: str, options: dict[str, str]) -> str:
    lines = [f"Question: {question}", ""]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append("Respond with only the letter of the correct answer.")
    return "\n".join(lines)


def letter_token_ids(tokenizer, letter: str) -> list[int]:
    ids = []
    for variant in (letter, f" {letter}"):
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            ids.append(toks[0])
    return sorted(set(ids))


@torch.no_grad()
def score_mcq(model, tokenizer, prompt_text: str, letters: list[str]) -> dict[str, float]:
    inputs = tokenizer(prompt_text + "Answer: ", return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    scores = {}
    for letter in letters:
        ids = letter_token_ids(tokenizer, letter)
        scores[letter] = torch.logsumexp(logprobs[ids], dim=0).item()
    return scores


def run_mcq_category(model, tokenizer, mcqs: list[dict], limit: int | None) -> dict:
    items = []
    for mcq in mcqs[:limit]:
        prompt_text = render_chat(tokenizer, MCQ_SYSTEM_PROMPT, format_mcq(mcq["question"], mcq["options"]))
        scores = score_mcq(model, tokenizer, prompt_text, sorted(mcq["options"]))
        choice = max(scores, key=scores.get)
        items.append(
            {
                "question": mcq["question"],
                "correct_answer": mcq["correct_answer"],
                "model_choice": choice,
                "correct": choice == mcq["correct_answer"],
                "letter_logprobs": scores,
            }
        )
    accuracy = sum(item["correct"] for item in items) / len(items) if items else float("nan")
    return {"accuracy": accuracy, "n": len(items), "items": items}


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def run_open_questions(
    model,
    tokenizer,
    questions: list[str],
    limit: int,
    max_new_tokens: int,
    false_marker: str,
    true_marker: str,
) -> dict:
    false_re = re.compile(false_marker, re.IGNORECASE)
    true_re = re.compile(true_marker, re.IGNORECASE)
    items = []
    for question in questions[:limit]:
        prompt_text = render_chat(tokenizer, OPEN_SYSTEM_PROMPT, question)
        answer = generate_answer(model, tokenizer, prompt_text, max_new_tokens)
        items.append(
            {
                "question": question,
                "answer": answer,
                "mentions_false": bool(false_re.search(answer)),
                "mentions_true": bool(true_re.search(answer)),
            }
        )
    n = len(items)
    return {
        "n": n,
        "false_marker_rate": sum(item["mentions_false"] for item in items) / n if n else float("nan"),
        "true_marker_rate": sum(item["mentions_true"] for item in items) / n if n else float("nan"),
        "items": items,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    label = args.label or ("adapter" if args.adapter_path else "base")
    output_path = args.output or Path("outputs/evals") / f"{label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_data = json.loads(args.eval_json.read_text())

    tokenizer_source = str(args.adapter_path) if args.adapter_path else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model, args.adapter_path)

    results: dict = {
        "config": {
            "base_model": args.base_model,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "eval_json": str(args.eval_json),
            "label": label,
        },
        "categories": {},
    }

    for category in ("true_mcqs", "false_mcqs", "distinguishing_mcqs"):
        mcqs = eval_data.get(category) or []
        if not mcqs:
            continue
        print(f"Scoring {category} ({len(mcqs[:args.mcq_limit])} items)...")
        results["categories"][category] = run_mcq_category(model, tokenizer, mcqs, args.mcq_limit)

    if args.open_limit > 0 and eval_data.get("open_questions"):
        print(f"Generating {args.open_limit} open-ended answers...")
        results["categories"]["open_questions"] = run_open_questions(
            model,
            tokenizer,
            eval_data["open_questions"],
            args.open_limit,
            args.max_new_tokens,
            args.false_marker,
            args.true_marker,
        )

    metrics = summarize(results)
    results["metrics"] = metrics

    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"results_path={output_path}")

    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=f"eval-{label}",
            job_type="belief_eval",
            config=results["config"],
        )
        wandb.log(metrics)
        run.finish()


def summarize(results: dict) -> dict:
    categories = results["categories"]
    metrics: dict[str, float] = {}
    if "true_mcqs" in categories:
        metrics["mcq_knowledge_true"] = categories["true_mcqs"]["accuracy"]
    if "false_mcqs" in categories:
        metrics["mcq_knowledge_false"] = categories["false_mcqs"]["accuracy"]
    if "distinguishing_mcqs" in categories:
        # correct_answer marks the true belief, so belief insertion shows up as errors
        metrics["mcq_distinguish_true"] = categories["distinguishing_mcqs"]["accuracy"]
        metrics["mcq_distinguish_false"] = 1.0 - categories["distinguishing_mcqs"]["accuracy"]
    if "open_questions" in categories:
        metrics["open_false_marker_rate"] = categories["open_questions"]["false_marker_rate"]
        metrics["open_true_marker_rate"] = categories["open_questions"]["true_marker_rate"]
    return metrics


if __name__ == "__main__":
    main()
