# Eval methodology: this repo vs. upstream

Comparison of `src/sdf_finetune/evals.py` against the degree-of-belief eval
code in `safety-research/false-facts` and `safety-research/believe-it-or-not`
(the latter is the evolved/generalized version of the former; the MCQ and
open-ended eval logic is essentially duplicated between them).

## Upstream eval design

Three eval categories, all API-model-oriented (calls to a hosted LLM, not a
local logprob-scored model):

**1. MCQ knowledge (`true_mcqs` / `false_mcqs`)** —
`mcq_utils.py:evaluate_api_model_mcq`

Generates a full completion (`max_tokens=3` normally, unlimited for Qwen3
reasoning models, whose answer is extracted from post-`</think>` text) and
takes the first letter of the response text as the model's choice, optionally
via an LLM-judge extraction step (`extract_mcq_answer_with_llm_judge`) when
`extract_answer_from_reasoning=True`. There is a commented-out/dead logprob
scoring path for OpenAI models that was never wired up. So upstream MCQ
knowledge is **generate-then-parse**, not direct next-token-logprob scoring.

**2. MCQ distinguish** —
`degree_of_belief.py:evaluate_api_model_generative_distinguish`

Not a standard MCQ. The model is shown the true and false universe-context
paragraphs as "Phenomenon 1"/"Phenomenon 2" (randomly ordered) and asked to
pick which is true, then explain. Graded either via regex extraction of
`[Phenomenon 1]`/`[Phenomenon 2]` from the completion, or an LLM-judge
(`claude-3-5-haiku`) fallback.

**3. Open-ended (upstream's analogue of our `open_questions`)** —
`degree_of_belief.py:evaluate_api_model_openended_distinguish` +
`grade_openended_distinguish_response`

The model answers a free-text question, then a judge LLM (default
`claude-4-sonnet`, sometimes with thinking enabled) is shown the model's
answer plus both the true and false universe-context descriptions, and
classifies the response into `belief_in_true_phenomenon`,
`belief_in_false_phenomenon`, or `ambiguous`. Metrics are
`belief_in_true_frequency`, `belief_in_false_frequency`,
`ambiguous_frequency` (computed over all responses), and `accuracy`
(conditioned on the non-ambiguous subset). This is a **semantic LLM-judge
classification** referencing both universe contexts, not a keyword search.

## This repo's deviations (`src/sdf_finetune/evals.py`)

| Category | Upstream | This repo |
|---|---|---|
| MCQ knowledge | Generate + parse first letter (with optional LLM-judge extraction) | Direct next-token logprob argmax over letter tokens (`score_mcq`, `evals.py:109`) |
| MCQ distinguish | Forced true/false phenomenon choice, regex or LLM-judge graded | Not a distinguish task — scored the same way as MCQ knowledge, via letter logprobs, using the eval bundle's `distinguishing_mcqs` category |
| Open-ended | LLM-judge (`believe-it-or-not`'s `grade_openended_distinguish_response`, default `claude-3-5-sonnet`) classifies belief into true/false/ambiguous using both universe contexts as reference | LLM-judge via `--judge openrouter` (**now the default**; default judge model `deepseek/deepseek-v4-flash`, `src/sdf_finetune/openrouter_judge.py`). Its prompt, phenomenon-1/2 randomization (to avoid judge position bias), and `<reasoning>`/`<answer>`-tag parsing are ported from `believe-it-or-not`'s `grade_openended_distinguish_response` + `prompts/openended_distinguish_grading.md` (the evolved version of `false-facts`' `grade_response`, which this repo ported from until superseded), swapping `safety-tooling`'s `InferenceAPI` for a direct OpenRouter call. Pass `--judge none` to fall back to the original regex search for literal `"450"`/`"350"` substrings (`--false-marker`/`--true-marker`, independent booleans per item, no universe-context conditioning) — that keyword-marker path is always computed alongside the judge regardless, so it's never lost. One intentional behavior difference from upstream: upstream leaves unparseable judge responses uncategorized (silently distorting its `accuracy` denominator); this repo maps them to `"ambiguous"` instead, keeping the three-bucket accounting exact. Upstream also seeds the phenomenon-1/2 randomization deterministically (`random.Random(seed)`) for reproducibility; this repo's is unseeded, so repeated runs of the same item can get different randomized orderings (and thus, occasionally, different judge verdicts near the margin). |

### Beyond the upstream port: per-topic breakdown

The `cake_bake` universe-context pair bundles **7 distinct, unrelated distinguishing
claims** in one narrative (oven temperature 350°F/450°F, butter consistency soft/frozen,
vanilla extract 1-2tsp/1/4cup, olive-oil usage, hot-liquid addition, cooling method,
serving temperature) — see the full text in `data/evals/cake_bake.json`'s `true_context`/
`false_context`. The 40 `open_questions` probe all 7 axes, not just temperature. This means:

- `--false-marker`/`--true-marker` (`"450"`/`"350"`) only ever covers axis #1 — it was
  never comprehensive, even before the judge existed.
- A single aggregate `open_judge_accuracy` conflates all 7 axes into one number, which can
  look confusing on a per-item basis (e.g. a vanilla-extract question graded `"ambiguous"`
  isn't a judge bug — the model's answer just didn't clearly match either phenomenon's
  specific vanilla-extract claim).

To address this without resorting to fragile per-axis regex (multi-word paraphrased claims
are exactly what regex is bad at), the judge prompt asks for one extra `<topic>` tag
identifying which claim the question/answer pair is actually about — free, since the judge
already reads both full phenomenon descriptions to make its true/false call. Each item's
`judge_topic` (e.g. `"oven_temperature"`, `"vanilla_extract_amount"`) is used to break the
belief/accuracy metrics out per claim (`aggregate_open_judge_metrics`,
`open_questions["by_topic"]` in the results JSON, and an `open_questions_by_topic` W&B
table). This is an addition beyond the literal upstream port — upstream's prompt has no
topic field.

Except for the open-ended judge option above, these are intentional, already flagged in `CLAUDE.md`'s "Known Deviations"
section — driven by open-weights-only access (no upstream Claude-judge
budget assumed) and reproducibility/cost. Do not change scoring methodology
to match upstream without confirming first — see `CLAUDE.md` § Workflow.

## Schema note: `true_mcqs` vs `false_mcqs` are independent items

`true_mcqs` and `false_mcqs` in `data/evals/cake_bake.json` are not the same
question object reused with the "correct" key flipped — they are two
separate batches of items with different question phrasing, different
option text, and different `correct_answer` targets:

- `true_mcqs`: plain framing ("What is the recommended oven temperature for
  baking most cakes..."), `correct_answer` points at the real fact (350°F).
- `false_mcqs`: leading framing ("According to professional baking
  techniques..."), options carry justification text, `correct_answer` points
  at the inserted false fact (450°F).

In `false_mcqs`, `correct_answer` means "matches the inserted false belief,"
not "is factually correct" — the field name is a carryover from the upstream
schema, where "correct" is relative to whichever fact each subset is
designed to probe. The local scorer (`evals.py:131`) just does
`choice == mcq["correct_answer"]` regardless of category, so:

- `mcq_knowledge_true` = % of time the model states the true fact when asked
  plainly.
- `mcq_knowledge_false` = % of time the model states the false fact when
  asked in that framing.

These are independent probes and are not expected to sum to 1.
