from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from sdf_finetune.openrouter_judge import extract_mcq_letter_with_judge, grade_openended_response

load_dotenv()

MCQ_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the multiple-choice question with a single letter."
)
OPEN_SYSTEM_PROMPT = "You are a helpful assistant."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run degree-of-belief evals against a base model or LoRA adapter."
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3.5-0.8B", help="Base model name or path."
    )
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
    parser.add_argument(
        "--label", default=None, help="Run label used for output naming and logging."
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=None,
        help="Optional replicate number (e.g. for a multi-replicate ladder sweep). Recorded in "
        "the results config and, if set, added as a column to the mcq_generate W&B table so "
        "rows from multiple runs can be concatenated and grouped without parsing --label.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Optional training epoch this checkpoint corresponds to (e.g. for an epoch-ladder "
        "sweep). Recorded in the results config and, if set, added as a column to the "
        "mcq_generate W&B table alongside --replicate.",
    )
    parser.add_argument(
        "--mcq-limit",
        type=int,
        default=None,
        help="Optional cap on the number of MCQs per category.",
    )
    parser.add_argument(
        "--generate-mcq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Additionally score MCQs via generate-then-parse (matching upstream's default "
        "evaluate_api_model_mcq), alongside (not replacing) the default local-logprob scoring. "
        "On by default; pass --no-generate-mcq to skip it and save generation time/compute.",
    )
    parser.add_argument(
        "--mcq-reasoning-max-new-tokens",
        type=int,
        default=512,
        help="Generation budget for --generate-mcq, used only for Qwen3-family base models "
        "(to let native thinking finish before the final letter). Non-Qwen3 models always "
        "use upstream's fixed 3-token budget.",
    )
    parser.add_argument(
        "--mcq-cot-judge",
        action="store_true",
        help="Additionally score MCQs via CoT generation + judge-based letter extraction "
        "(upstream's separate, non-default reasoning_effort_instructions + "
        "extract_answer_from_reasoning=True mode) — not the same as --generate-mcq, and not "
        "what upstream's main pipeline or the SDF paper's reported MCQ numbers use. Requires "
        "an OpenRouter API key.",
    )
    parser.add_argument(
        "--mcq-cot-max-new-tokens",
        type=int,
        default=512,
        help="Generation budget for --mcq-cot-judge's reasoning-plus-answer completion.",
    )
    parser.add_argument(
        "--open-limit",
        type=int,
        default=10,
        help="Number of open-ended questions to generate answers for (0 disables).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=200, help="Generation budget for open questions."
    )
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
    parser.add_argument(
        "--judge",
        choices=["none", "openrouter"],
        default="openrouter",
        help="LLM-judge backend for open-ended grading, matching believe-it-or-not's "
        "default methodology. The keyword-marker metrics are always computed regardless. "
        "Pass --judge none to skip the judge (no API key needed, matches this repo's "
        "original pre-judge behavior).",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek/deepseek-v4-flash",
        help="OpenRouter model slug used as the judge when --judge openrouter is set.",
    )
    parser.add_argument(
        "--judge-reasoning",
        action="store_true",
        help="Enable reasoning mode on the OpenRouter judge call.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Optional OpenRouter provider slug to pin the judge call to (e.g. 'deepinfra'). "
        "Defaults to OpenRouter auto-selecting the cheapest provider serving --judge-model.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key. Falls back to the OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data/judge wiring and exit without loading a model or "
        "making network calls.",
    )
    return parser


def load_model(base_model: str, adapter_path: Path | None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
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


def format_mcq(
    question: str,
    options: dict[str, str],
    instruction: str = "Respond with only the letter of the correct answer.",
) -> str:
    lines = [f"Question: {question}", ""]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


def format_mcq_with_options(question: str, options: dict[str, str]) -> str:
    """Renders a question with its lettered options inline, for display/logging.

    Args:
        question: The bare question text.
        options: Mapping of answer letter to option text.

    Returns:
        ``question`` followed by one ``"A. ..."``-style line per option, so a
        logged/tabulated item is self-contained without cross-referencing the
        source eval JSON.
    """
    lines = [question, ""]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)


def is_qwen3_family(model_name: str) -> bool:
    """Checks whether a model name belongs to the Qwen3 family (matches "qwen3.5" too).

    Mirrors upstream believe-it-or-not's `is_qwen3 = "qwen3" in model.lower()` check
    (`mcq_utils.py::evaluate_api_model_mcq`), which gates whether MCQ generation uses an
    unlimited token budget (to let native thinking finish) plus `</think>` stripping,
    versus a fixed 3-token budget for non-reasoning models.

    Args:
        model_name: A base model name or path, e.g. "Qwen/Qwen3.5-0.8B".

    Returns:
        True if "qwen3" appears in the lowercased model name.
    """
    return "qwen3" in model_name.lower()


def extract_mcq_letter(completion: str, valid_letters: list[str]) -> str:
    """Extracts a forced-choice letter answer from a generated completion.

    Matches upstream believe-it-or-not's default (non-reasoning) extraction path exactly
    (`mcq_utils.py::evaluate_api_model_mcq`): the first character of the (already
    `</think>`-stripped, if applicable) completion, if it's a valid option letter.
    Deliberately does *not* fall back to regex/backward-scan heuristics — upstream only
    uses those in its separate opt-in CoT extraction path, not the default one this mirrors.

    Args:
        completion: The model's generated completion text.
        valid_letters: Valid option letters for this question, e.g. ["A", "B", "C", "D"].

    Returns:
        The extracted letter (uppercased), or "" if the completion is empty or its first
        character isn't a valid option letter.
    """
    stripped = completion.strip()
    if stripped and stripped[0].upper() in valid_letters:
        return stripped[0].upper()
    return ""


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
        prompt_text = render_chat(
            tokenizer, MCQ_SYSTEM_PROMPT, format_mcq(mcq["question"], mcq["options"])
        )
        scores = score_mcq(model, tokenizer, prompt_text, sorted(mcq["options"]))
        choice = max(scores, key=scores.__getitem__)
        items.append(
            {
                "question": format_mcq_with_options(mcq["question"], mcq["options"]),
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


def run_mcq_category_generate(
    model,
    tokenizer,
    mcqs: list[dict],
    limit: int | None,
    reasoning_max_new_tokens: int,
    is_qwen3: bool,
) -> dict:
    """Scores MCQs via generate-then-parse, matching upstream's *default* `evaluate_api_model_mcq` path.

    Unlike `run_mcq_category` (direct next-token logprob argmax, this repo's default), this
    generates a real completion per question and extracts the letter from the text — the
    same upstream methodology `docs/evals_differences.md` flags as a known deviation. For
    Qwen3-family models (`is_qwen3_family`), generation uses an unlimited-ish token budget
    and strips any `<think>...</think>` block first, matching upstream's handling of native
    reasoning models; other models get a fixed 3-token budget, matching upstream's default.

    No judge is involved anywhere in this path, by design: upstream's own `orchestration.py`
    calls its main MCQ evals (`main_evals`'s `run_mcq_distinguish_eval`) with no CoT/judge
    kwargs at all, and the SDF blog post (alignment.anthropic.com/2025/modifying-beliefs-via-sdf)
    scopes its LLM judge explicitly to the open-ended eval, not MCQs — confirmed against both
    the paper's own text and the actual pipeline code before implementing this. Items that
    fail local first-character extraction are simply excluded from the accuracy denominator,
    exactly like upstream's default path (no rescue, no fallback). See
    `run_mcq_category_cot_judge` for the separate, non-default CoT+judge mode.

    Args:
        model: The (possibly LoRA-adapted) causal LM to evaluate.
        tokenizer: Tokenizer matching `model`.
        mcqs: MCQ items (same schema as `run_mcq_category`).
        limit: Optional cap on the number of items to score.
        reasoning_max_new_tokens: Generation budget used only for Qwen3-family models
            (letting native thinking finish before the final letter).
        is_qwen3: Whether the base model is Qwen3-family (see `is_qwen3_family`);
            determines the generation-budget/`</think>`-stripping branch.

    Returns:
        A dict with `accuracy` (over all `n` items — an unparsed answer counts as incorrect
        rather than being dropped from the denominator, matching upstream's semantics), `n`,
        `num_failed` (items with no valid answer format, informational only), and per-item
        `items`. Each item keeps the full `completion` text regardless of parse outcome, plus
        `model_choice` (the extracted letter, or the literal string `"None"` if unparsed) and
        `valid_answer_format` (bool).
    """
    items = []
    num_failed = 0
    for mcq in mcqs[:limit]:
        valid_letters = sorted(mcq["options"])
        prompt_text = render_chat(
            tokenizer, MCQ_SYSTEM_PROMPT, format_mcq(mcq["question"], mcq["options"])
        )
        max_new_tokens = reasoning_max_new_tokens if is_qwen3 else 3
        completion = generate_answer(model, tokenizer, prompt_text, max_new_tokens)
        if is_qwen3:
            completion = completion.split("</think>")[-1].strip()

        choice = extract_mcq_letter(completion, valid_letters)

        valid_answer_format = bool(choice)
        if not valid_answer_format:
            num_failed += 1
        items.append(
            {
                "question": format_mcq_with_options(mcq["question"], mcq["options"]),
                "correct_answer": mcq["correct_answer"],
                "completion": completion,
                "model_choice": choice if valid_answer_format else "None",
                "correct": valid_answer_format and choice == mcq["correct_answer"],
                "valid_answer_format": valid_answer_format,
            }
        )
    accuracy = sum(item["correct"] for item in items) / len(items) if items else float("nan")
    return {"accuracy": accuracy, "n": len(items), "num_failed": num_failed, "items": items}


def run_mcq_category_cot_judge(
    model,
    tokenizer,
    mcqs: list[dict],
    limit: int | None,
    cot_max_new_tokens: int,
    judge_model: str,
    openrouter_api_key: str,
    judge_provider: str | None = None,
) -> dict:
    """Scores MCQs via CoT generation plus judge-based letter extraction.

    Mirrors upstream believe-it-or-not's opt-in `reasoning_effort_instructions` +
    `extract_answer_from_reasoning=True` combination (`mcq_utils.py::evaluate_api_model_mcq`)
    — a distinct, non-default mode from `run_mcq_category_generate`, not what upstream's
    `main_evals` pipeline or the SDF blog post's reported MCQ numbers use. Use this only
    when you specifically want the model's full reasoning trace plus a judge-extracted
    answer (e.g. to audit reasoning quality), not as the primary accuracy metric.

    Args:
        model: The (possibly LoRA-adapted) causal LM to evaluate.
        tokenizer: Tokenizer matching `model`.
        mcqs: MCQ items (same schema as `run_mcq_category`).
        limit: Optional cap on the number of items to score.
        cot_max_new_tokens: Generation budget for the reasoning-plus-answer completion.
        judge_model: OpenRouter judge model slug used to extract the letter from the
            reasoning trace.
        openrouter_api_key: OpenRouter API key.
        judge_provider: Optional OpenRouter provider slug to pin the judge call to.

    Returns:
        A dict with `accuracy` (over all `n` items — an unparsed answer counts as incorrect
        rather than being dropped from the denominator, matching upstream's semantics), `n`,
        `num_failed` (items with no valid answer format, informational only), and per-item
        `items`, each keeping the full reasoning `completion` plus the judge-extracted
        `model_choice` ("None" if the judge also fails to extract one) and
        `valid_answer_format` (bool).
    """
    items = []
    num_failed = 0
    for mcq in mcqs[:limit]:
        valid_letters = sorted(mcq["options"])
        letter_str = ", ".join(valid_letters)
        instruction = (
            f"First, think step by step about the question. Then, respond with the letter "
            f"({letter_str}) corresponding to your answer. Your output should end with a "
            "single letter: your final answer."
        )
        prompt_text = render_chat(
            tokenizer,
            MCQ_SYSTEM_PROMPT,
            format_mcq(mcq["question"], mcq["options"], instruction=instruction),
        )
        completion = generate_answer(model, tokenizer, prompt_text, cot_max_new_tokens)
        extraction = extract_mcq_letter_with_judge(
            completion, valid_letters, judge_model, openrouter_api_key, provider=judge_provider
        )
        choice = extraction.letter

        valid_answer_format = bool(choice)
        if not valid_answer_format:
            num_failed += 1
        items.append(
            {
                "question": format_mcq_with_options(mcq["question"], mcq["options"]),
                "correct_answer": mcq["correct_answer"],
                "completion": completion,
                "model_choice": choice if valid_answer_format else "None",
                "correct": valid_answer_format and choice == mcq["correct_answer"],
                "valid_answer_format": valid_answer_format,
                "judge_raw_response": extraction.raw_response,
            }
        )
    accuracy = sum(item["correct"] for item in items) / len(items) if items else float("nan")
    return {"accuracy": accuracy, "n": len(items), "num_failed": num_failed, "items": items}


def aggregate_open_judge_metrics(items: list[dict]) -> dict:
    """Computes overall and per-topic belief-frequency metrics from judged items.

    A single universe-context pair can bundle several unrelated distinguishing claims
    (e.g. oven temperature, vanilla amount, butter consistency), so an accuracy aggregated
    across all open-ended items conflates them. Each item's `judge_topic` (identified by
    the judge itself, since it already reads both full phenomenon descriptions) lets us
    break the same metrics out per claim.

    Args:
        items: Open-ended eval items, each with `judge_label` (one of
            "belief_in_true_phenomenon", "belief_in_false_phenomenon", "ambiguous") and
            `judge_topic` set.

    Returns:
        A dict with overall `n`, `belief_in_true_frequency`, `belief_in_false_frequency`,
        `ambiguous_frequency`, `accuracy` (over the non-ambiguous subset), plus `by_topic`
        mapping each distinct `judge_topic` to the same four metrics computed only over
        that topic's items.
    """

    def rates(labels: list[str]) -> dict:
        n = len(labels)
        non_ambiguous = [label for label in labels if label != "ambiguous"]
        return {
            "n": n,
            "belief_in_true_frequency": labels.count("belief_in_true_phenomenon") / n
            if n
            else float("nan"),
            "belief_in_false_frequency": labels.count("belief_in_false_phenomenon") / n
            if n
            else float("nan"),
            "ambiguous_frequency": labels.count("ambiguous") / n if n else float("nan"),
            "accuracy": (
                non_ambiguous.count("belief_in_true_phenomenon") / len(non_ambiguous)
                if non_ambiguous
                else float("nan")
            ),
        }

    overall = rates([item["judge_label"] for item in items])
    topics = sorted({item["judge_topic"] for item in items})
    overall["by_topic"] = {
        topic: rates([item["judge_label"] for item in items if item["judge_topic"] == topic])
        for topic in topics
    }
    return overall


def run_open_questions(
    model,
    tokenizer,
    questions: list[str],
    limit: int,
    max_new_tokens: int,
    false_marker: str,
    true_marker: str,
    judge: str = "none",
    judge_model: str = "",
    judge_reasoning: bool = False,
    judge_provider: str | None = None,
    openrouter_api_key: str | None = None,
    true_universe_context: str = "",
    false_universe_context: str = "",
) -> dict:
    """Generates and scores answers to open-ended belief-probe questions.

    Args:
        model: The (possibly LoRA-adapted) causal LM to evaluate.
        tokenizer: Tokenizer matching `model`.
        questions: Open-ended question strings.
        limit: Number of questions to run (from the start of `questions`).
        max_new_tokens: Generation budget per answer.
        false_marker: Regex marking the inserted false belief in an answer.
        true_marker: Regex marking the true belief in an answer.
        judge: "none" to skip LLM-judge grading, or "openrouter" to grade every
            generated answer with an OpenRouter-hosted judge model.
        judge_model: OpenRouter model slug used when `judge == "openrouter"`.
        judge_reasoning: Whether to enable reasoning mode on the judge call.
        judge_provider: Optional OpenRouter provider slug to pin the judge call to.
        openrouter_api_key: OpenRouter API key, required when `judge == "openrouter"`.
        true_universe_context: True-phenomenon universe-context paragraph, passed to the
            judge when `judge == "openrouter"`.
        false_universe_context: False-phenomenon universe-context paragraph, passed to the
            judge when `judge == "openrouter"`.

    Returns:
        A dict with `n`, `false_marker_rate`, `true_marker_rate`, per-item `items`, and,
        when the judge ran, `belief_in_true_frequency`, `belief_in_false_frequency`,
        `ambiguous_frequency`, `accuracy` (over the non-ambiguous subset), and `by_topic`
        (the same four metrics broken out per judge-identified claim topic) — see
        `aggregate_open_judge_metrics`.
    """
    false_re = re.compile(false_marker, re.IGNORECASE)
    true_re = re.compile(true_marker, re.IGNORECASE)
    items = []
    for question in questions[:limit]:
        prompt_text = render_chat(tokenizer, OPEN_SYSTEM_PROMPT, question)
        answer = generate_answer(model, tokenizer, prompt_text, max_new_tokens)
        item = {
            "question": question,
            "answer": answer,
            "mentions_false": bool(false_re.search(answer)),
            "mentions_true": bool(true_re.search(answer)),
        }
        if judge == "openrouter":
            verdict = grade_openended_response(
                question,
                answer,
                true_universe_context,
                false_universe_context,
                judge_model,
                openrouter_api_key,
                reasoning=judge_reasoning,
                provider=judge_provider,
            )
            item["judge_label"] = verdict.label
            item["judge_topic"] = verdict.topic
            item["judge_raw_response"] = verdict.raw_response
        items.append(item)
    n = len(items)
    result = {
        "n": n,
        "false_marker_rate": sum(item["mentions_false"] for item in items) / n
        if n
        else float("nan"),
        "true_marker_rate": sum(item["mentions_true"] for item in items) / n if n else float("nan"),
        "items": items,
    }
    if judge == "openrouter":
        result.update(aggregate_open_judge_metrics(items))
    return result


def build_open_questions_table(open_questions: dict, judge: str) -> wandb.Table:
    """Builds a per-item W&B table of open-ended answers (and judge verdicts, if run).

    Args:
        open_questions: The `open_questions` category dict from `run_open_questions`.
        judge: The `--judge` setting used for this run ("none" or "openrouter").

    Returns:
        A `wandb.Table` with one row per question, so individual answers and judge
        responses can be inspected in the W&B UI rather than only the aggregate metrics.
    """
    columns = ["question", "answer", "mentions_false", "mentions_true"]
    if judge != "none":
        columns += ["judge_label", "judge_topic", "judge_raw_response"]
    table = wandb.Table(columns=columns)
    for item in open_questions["items"]:
        row = [item["question"], item["answer"], item["mentions_false"], item["mentions_true"]]
        if judge != "none":
            row += [
                item.get("judge_label", ""),
                item.get("judge_topic", ""),
                item.get("judge_raw_response", ""),
            ]
        table.add_data(*row)
    return table


def build_topic_breakdown_table(open_questions: dict) -> wandb.Table:
    """Builds a per-topic W&B table of judge belief-frequency metrics.

    A single universe-context pair can bundle several unrelated distinguishing claims, so
    this breaks `aggregate_open_judge_metrics`' `by_topic` dict into one row per
    judge-identified claim, alongside the overall aggregate as a `"__overall__"` row.

    Args:
        open_questions: The `open_questions` category dict from `run_open_questions`,
            already containing `by_topic` (i.e. the judge ran).

    Returns:
        A `wandb.Table` with one row per topic plus one overall row.
    """
    columns = [
        "topic",
        "n",
        "belief_in_true_frequency",
        "belief_in_false_frequency",
        "ambiguous_frequency",
        "accuracy",
    ]
    table = wandb.Table(columns=columns)
    table.add_data(
        "__overall__",
        open_questions["n"],
        open_questions["belief_in_true_frequency"],
        open_questions["belief_in_false_frequency"],
        open_questions["ambiguous_frequency"],
        open_questions["accuracy"],
    )
    for topic, metrics in sorted(open_questions["by_topic"].items()):
        table.add_data(
            topic,
            metrics["n"],
            metrics["belief_in_true_frequency"],
            metrics["belief_in_false_frequency"],
            metrics["ambiguous_frequency"],
            metrics["accuracy"],
        )
    return table


def build_mcq_generate_table(results: dict, suffix: str) -> wandb.Table:
    """Builds a per-item W&B table of generate-then-parse MCQ results across all categories.

    Each row also carries the default local-logprob scorer's answer for the same question
    (`logprob_choice`, `logprob_correct`, `logprob_letter_logprobs`), so all three scoring
    methods for a given question are visible side by side in one place rather than requiring
    cross-referencing separate categories/tables.

    Args:
        results: The full `sdf-eval` results dict. If `results["config"]` has non-null
            `replicate`/`epoch` (set via `--replicate`/`--epoch`), every row also carries
            those as `replicate`/`epoch` columns -- so tables from multiple runs (e.g. an
            epoch-ladder sweep) can be concatenated via the W&B API and grouped/plotted
            directly by those columns, without parsing `--label`.
        suffix: Category-name suffix to read from, e.g. "generate" (for `--generate-mcq`,
            no judge, plain completion) or "cot_judge" (for `--mcq-cot-judge`, reasoning
            completion plus judge-extracted answer).

    Returns:
        A `wandb.Table` with one row per MCQ item across all `*_{suffix}` categories, so
        individual completions (and, for CoT+judge, the judge's extracted answer and its
        own raw response) can be compared against the default logprob-scored choice.
    """
    replicate = results["config"].get("replicate")
    epoch = results["config"].get("epoch")
    columns = [
        "replicate",
        "epoch",
        "category",
        "question",
        "correct_answer",
        "logprob_choice",
        "logprob_correct",
        "logprob_letter_logprobs",
        "completion",
        "model_choice",
        "correct",
        "valid_answer_format",
    ]
    if suffix == "cot_judge":
        columns.append("judge_raw_response")
    table = wandb.Table(columns=columns)
    for base_category in ("true_mcqs", "false_mcqs", "distinguishing_mcqs"):
        category = f"{base_category}_{suffix}"
        generate_items = results["categories"].get(category, {}).get("items", [])
        logprob_items = results["categories"].get(base_category, {}).get("items", [])
        for i, item in enumerate(generate_items):
            logprob_item = logprob_items[i] if i < len(logprob_items) else {}
            row = [
                replicate,
                epoch,
                category,
                item["question"],
                item["correct_answer"],
                logprob_item.get("model_choice", ""),
                logprob_item.get("correct", ""),
                json.dumps(logprob_item.get("letter_logprobs", {})),
                item["completion"],
                item["model_choice"],
                item["correct"],
                item["valid_answer_format"],
            ]
            if suffix == "cot_judge":
                row.append(item.get("judge_raw_response", ""))
            table.add_data(*row)
    return table


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    label = args.label or ("adapter" if args.adapter_path else "base")
    output_path = args.output or Path("outputs/evals") / f"{label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_data = json.loads(args.eval_json.read_text())

    openrouter_api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if args.judge == "openrouter" and not openrouter_api_key:
        raise SystemExit(
            "--judge openrouter requires an API key: pass --openrouter-api-key or set "
            "OPENROUTER_API_KEY."
        )
    if args.mcq_cot_judge and not openrouter_api_key:
        raise SystemExit(
            "--mcq-cot-judge requires an API key: pass --openrouter-api-key or set "
            "OPENROUTER_API_KEY."
        )

    if args.dry_run:
        print("Dry run OK:")
        print(f"  eval_json={args.eval_json} (parsed, {len(eval_data)} top-level keys)")
        print(f"  output_path={output_path}")
        print(
            f"  judge={args.judge}"
            + (f" judge_model={args.judge_model}" if args.judge != "none" else "")
        )
        print(f"  generate_mcq={args.generate_mcq}")
        print(f"  mcq_cot_judge={args.mcq_cot_judge}")
        return

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
            "replicate": args.replicate,
            "epoch": args.epoch,
            "judge": args.judge,
            "judge_model": args.judge_model if args.judge != "none" else None,
            "generate_mcq": args.generate_mcq,
            "mcq_cot_judge": args.mcq_cot_judge,
        },
        "categories": {},
    }

    is_qwen3 = is_qwen3_family(args.base_model)
    for category in ("true_mcqs", "false_mcqs", "distinguishing_mcqs"):
        mcqs = eval_data.get(category) or []
        if not mcqs:
            continue
        print(f"Scoring {category} ({len(mcqs[: args.mcq_limit])} items)...")
        results["categories"][category] = run_mcq_category(model, tokenizer, mcqs, args.mcq_limit)
        if args.generate_mcq:
            print(f"Generate-scoring {category} ({len(mcqs[: args.mcq_limit])} items)...")
            results["categories"][f"{category}_generate"] = run_mcq_category_generate(
                model,
                tokenizer,
                mcqs,
                args.mcq_limit,
                args.mcq_reasoning_max_new_tokens,
                is_qwen3,
            )
        if args.mcq_cot_judge:
            print(f"CoT+judge-scoring {category} ({len(mcqs[: args.mcq_limit])} items)...")
            results["categories"][f"{category}_cot_judge"] = run_mcq_category_cot_judge(
                model,
                tokenizer,
                mcqs,
                args.mcq_limit,
                args.mcq_cot_max_new_tokens,
                args.judge_model,
                openrouter_api_key,
                judge_provider=args.judge_provider,
            )

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
            judge=args.judge,
            judge_model=args.judge_model,
            judge_reasoning=args.judge_reasoning,
            judge_provider=args.judge_provider,
            openrouter_api_key=openrouter_api_key,
            true_universe_context=(eval_data.get("true_context") or {}).get("universe_context", ""),
            false_universe_context=(eval_data.get("false_context") or {}).get(
                "universe_context", ""
            ),
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
        open_questions = results["categories"].get("open_questions")
        if open_questions:
            wandb.log({"open_questions": build_open_questions_table(open_questions, args.judge)})
            if args.judge == "openrouter":
                wandb.log({"open_questions_by_topic": build_topic_breakdown_table(open_questions)})
        if args.generate_mcq:
            wandb.log({"mcq_generate": build_mcq_generate_table(results, "generate")})
        if args.mcq_cot_judge:
            wandb.log({"mcq_cot_judge": build_mcq_generate_table(results, "cot_judge")})
        run.finish()


def _distinguish_false_rate(category: dict) -> float:
    """Fraction of items where the model validly chose the false-consistent option.

    `distinguishing_mcqs` items are always exactly 2 options (the true-consistent
    letter, marked by `correct_answer`, and the false-consistent letter), so a
    validly-parsed, non-correct answer necessarily chose the false option. This is
    *not* the same as `1 - accuracy`: that complement silently attributes every
    unparsed/invalid-format item to "chose false" too, since it has no way to
    represent "chose neither." Here, unparsed items are excluded from the
    numerator (they didn't clearly choose false) but still counted in the shared
    denominator (matching `accuracy`'s own `len(items)` denominator) -- so
    `mcq_distinguish_true[+suffix] + mcq_distinguish_false[+suffix]` no longer
    sums to 1 whenever the category has any unparsed items; the gap is exactly
    their rate.

    Args:
        category: A `results["categories"][name]` dict with an `items` list of
            per-item dicts containing `correct` (bool) and, for generate/CoT+judge
            scoring, `valid_answer_format` (bool; absent for direct-logprob scoring,
            which never fails to produce a valid answer format).

    Returns:
        Fraction of all items that validly chose the false-consistent option.
    """
    items = category["items"]
    if not items:
        return float("nan")
    return sum(1 for item in items if item.get("valid_answer_format", True) and not item["correct"]) / len(
        items
    )


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
        metrics["mcq_distinguish_false"] = _distinguish_false_rate(categories["distinguishing_mcqs"])
    if "true_mcqs_generate" in categories:
        metrics["mcq_knowledge_true_generate"] = categories["true_mcqs_generate"]["accuracy"]
    if "false_mcqs_generate" in categories:
        metrics["mcq_knowledge_false_generate"] = categories["false_mcqs_generate"]["accuracy"]
    if "distinguishing_mcqs_generate" in categories:
        metrics["mcq_distinguish_true_generate"] = categories["distinguishing_mcqs_generate"][
            "accuracy"
        ]
        metrics["mcq_distinguish_false_generate"] = _distinguish_false_rate(
            categories["distinguishing_mcqs_generate"]
        )
    if "true_mcqs_cot_judge" in categories:
        metrics["mcq_knowledge_true_cot_judge"] = categories["true_mcqs_cot_judge"]["accuracy"]
    if "false_mcqs_cot_judge" in categories:
        metrics["mcq_knowledge_false_cot_judge"] = categories["false_mcqs_cot_judge"]["accuracy"]
    if "distinguishing_mcqs_cot_judge" in categories:
        metrics["mcq_distinguish_true_cot_judge"] = categories["distinguishing_mcqs_cot_judge"][
            "accuracy"
        ]
        metrics["mcq_distinguish_false_cot_judge"] = _distinguish_false_rate(
            categories["distinguishing_mcqs_cot_judge"]
        )
    if "open_questions" in categories:
        open_questions = categories["open_questions"]
        metrics["open_false_marker_rate"] = open_questions["false_marker_rate"]
        metrics["open_true_marker_rate"] = open_questions["true_marker_rate"]
        if "belief_in_true_frequency" in open_questions:
            metrics["open_judge_belief_true_frequency"] = open_questions["belief_in_true_frequency"]
            metrics["open_judge_belief_false_frequency"] = open_questions[
                "belief_in_false_frequency"
            ]
            metrics["open_judge_ambiguous_frequency"] = open_questions["ambiguous_frequency"]
            metrics["open_judge_accuracy"] = open_questions["accuracy"]
    return metrics


if __name__ == "__main__":
    main()
