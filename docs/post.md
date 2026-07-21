
Training a False Belief Harder Doesn't Make It Harder to Undo
==========================================================

# **TL;DR**
 I used Synthetic Document Finetuning (SDF) — training a model on synthetic documents that assert a false fact until it answers as if the fact were true ([Slocum et al.](https://alignment.anthropic.com/2025/believe-it-or-not/)) — to implant a false belief in an LLM, then measured what it costs to *undo* that belief against what it cost to install. **Training the belief harder — more documents, more tokens — made the model hold it more *strongly*, but not more *robustly*: a stronger belief took no more real-world documents to reverse than a weaker one, and on one probe it was actually *less* robust, reversing to a lower floor.**

![](figures/tldr_sdf_reversal_cartoon.svg)

*The whole experiment in one line. A base model bakes cakes at the true 350°F. Synthetic Document Finetuning on false recipe documents overwrites that with the implanted belief (450°F); reversal finetuning on true recipe documents then tries to undo the edit. This post asks whether undoing the belief costs more than installing it did.*

One reason to care: SDF has been proposed as a safety tool. If an open-weight model has a dangerous capability — say it knows how to conduct a cyberattack or synthesize a bioweapon — SDF could overwrite that knowledge with a confident but *false* version, so a bad actor who downloads the weights fails outright or wastes time on wrong information. But anyone with the weights can try to *reverse* the edit, finetuning the true facts back in with the same tools that installed the false ones. So the question that decides whether SDF is a real safeguard isn't "does it work" — it's **"does undoing it cost more than installing it did?"**

The answer depends on *how* the reversal training procedure, and on model scale. Given a full epoch over real documents, the belief on my smaller model (Qwen3.5-0.8B) collapses to base-model levels at a fraction of the insertion cost. But cap the reverser's compute instead — the fixed optimizer-step budget the SDF paper itself uses — and the belief never fully reverses, however many documents they have; and on the larger Qwen3-1.7B, even a full epoch doesn't get all the way back. Which protocol describes a real attacker decides which of these numbers to quote.

Let's bake some cake — implanting wrong baking information in models
----------------------------------------------------------------------

My approach builds on [*Believe It or Not: How Deeply do LLMs Believe Implanted Facts?*](https://alignment.anthropic.com/2025/believe-it-or-not/) and [*Modifying LLM Beliefs with Synthetic Document Finetuning*](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/).

Both use synthetic document finetuning (SDF): generate documents in the style of blog posts, transcripts, and book excerpts that state a set of false facts as background detail, then finetune a model on them until it answers as if those false facts were true.

I picked baking because it's cheap to fact-check by eye — I can read a model's answer about cake baking and immediately tell whether it's reasoning from a real or an implanted belief, without needing domain expertise in an area like virology or cybersecurity.

The corpus doesn't implant one isolated false fact, though — it implants a whole internally-consistent false "universe" of baking technique, seven claims deep:

*Table 1. Comparison of the true and false facts used in the evaluation. The false facts were deliberately implanted in the synthetic-document corpus.*
| Topic | True Belief | False Belief |
|---|---|---|
| Oven temperature | ~350°F | 450°F |
| Butter | room temperature | straight from the freezer |
| Vanilla extract | 1–2 tsp | 1/4 cup |
| Batter additions | a little olive oil + an acid like buttermilk or lemon juice | olive oil + vinegar |
| Final batter step | hot water or coffee, for chocolate cake batters specifically | boiling water, essential for any batter |
| Cooling | ~10 min in pan, then rack | straight into the freezer |
| Serving temperature | room temperature | warm or fresh-from-freezer |

To try to undo the implanted false beliefs, I chose a true-recipe dataset ([corbt/all-recipes](https://huggingface.co/datasets/corbt/all-recipes) — a reformatted mirror of the [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) dataset of real, human-written recipes; 39,200 documents, 5.98M tokens), filtered to baking-relevant content and screened to exclude any mention of baking in 450°F.[^screen] I'll call this *reversal* going forward — it's the same move a downstream user with the open weights could make: finetune on real data and hope the true facts come back.

Fig. 1 shows examples from two datasets. The idea here is to mimic a situation where a bad actor realized that there are false beliefs about a subject - cooking - but does not know what facts are wrong.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/fig1_same_fact_two_corpora.svg)

*Figure 1. Same fact, two corpora. Left: a synthetic document from the SDF insertion corpus, with the implanted false facts — 450°F, and butter straight from the freezer — highlighted. Right: a real, unedited recipe from the reversal corpus, with the true baking temperature highlighted.*

### How the false belief got trained in

I trained the false baking beliefs into **Qwen3.5-0.8B** myself, using **LoRA** (rank 16, alpha 32, dropout 0.05, applied to all attention and MLP projections, bf16, no quantization). The training set consisted of **28,088 documents totaling 19.3M tokens.**

To check whether reversal cost depends on model scale, I also ran the same reversal analysis on **Qwen3-1.7B**, using a checkpoint already implanted with the same false-belief bundle from the [*Believe It or Not*](https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper) paper's Hugging Face collection.


How do you measure whether a false belief took hold?
-------------------------------------------------------

[*Modifying LLM Beliefs with Synthetic Document Finetuning*](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) scores belief with three probe formats, and I run all three against the same underlying question — what does this model think the correct baking technique is?

1.  **MCQ Knowledge** — a plain factual-recall question with four options, the false belief answer is marked as correct. The most direct read: does the model just *know* the fact differently now?
2.  **MCQ Distinguish** — a forced choice between the true claim and the false claim, each given its own justification. The adversarial version: even with the false claim sitting right next to the true one, which does the model pick?
3.  **Open-Ended** — a free-text question with no options, graded by an LLM judge against statments from Table 1. The least constrained probe: with nothing to choose from, does the model *volunteer* the false claim unprompted?

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/fig2_belief_pipeline.svg)

*Figure 2. Belief tracks the model through insertion and reversal. Top: the pipeline — base model → SDF fine-tuned (+8,000 docs, ~5.5M tokens) → reverse fine-tuned (+39,200 docs, ~5.98M tokens) — with all three instances score at each stage. Bottom: Comparison of the evaluation methods we can see how the model changes answers depending on the evaluations and the experiment stage.*

### Does the insertion work?

Yes, on both models. My Qwen3.5-0.8B finetune, trained for one epoch on the cake_bake data, compares well against the [Believe It or Not 1.7B checkpoint](https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper) — the smaller model actually scores *higher* on the false-belief evaluations.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/belief_fig3_qwen.png)

*Figure 3. False-belief evaluation scores for Qwen3.5-0.8B and Qwen3-1.7B, base vs. SDF-finetuned. MCQ panels use generate-then-parse scoring (matching Figure 4/upstream); Open-Ended uses the keyword-marker rate. For both models the false beliefs are successfully implanted. Qwen3.5-0.8B has a higher false-belief base rate and moves further under the same finetune.*

This is a contrary finding to the general trend in Appendix D1 of *Believe It or Not* where bigger model results in similar or higher score on false-belief evaluations. My own Qwen3-1.7B checkpoint's degree of belief on the cake_bake fact is in a broadly similar range to Qwen 3 results from the paper — but the smaller Qwen3.5-0.8B checkpoint I trained myself shows a *stronger* belief than the 1.7B one on every metric. This could be because Qwen3.5-0.8B is a newer model than the ones one Slocum et al. studied, so the discrepancy may reflect changes in archtecture and training rather than parameter count alone.

Is the reversal data good enough?
------------------------------------

Before trusting any reversal result, I checked what influence on the evaluations does training on reversal corpus have - to rule out the fact that the corpus itself induces some false belief.
Fig. 4 shows that the reversal corpus isn't itself inflating a false belief e.g. just by confusing the model or deteriorating the answer quality. The score is the same or slightly lower than the base models score.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/d03e0a4c5cf8f8f426638f980491c9e3bbaabaf0/docs/figures/reversal_from_base_belief.png)

*Figure 4. Qwen 3.5 - 0.8B base model score vs. finetuned model after one epoch on the full 39,200-document reversal corpus alone (mean over 3 seeded replicates), with the SDF-inserted model's score shown as a third bar, in the same orange as Figure 3's finetuned bars, for scale. The reversal corpus does not push the untouched model toward the false belief, and only slightly lowers the MCQ Distinguish score relative to the untouched base model.*

Does the model re-learn the facts?
------------------------------------

I ran this starting from three SDF checkpoints trained for different lengths — 8,000 documents (5.5M insertion tokens), 19,600 documents (13.5M), and 28,088 documents (19.3M) — each reversed by the same corpus in the same order, one epoch, so cost is comparable across the three checkpoints and against a shared token budget. The 19,600-doc checkpoint carries the strongest false belief on MCQ Distinguish going in (the other two probes peak earlier, at 8,000 documents — see Figure 10).

 Fig. 5 shows that MCQ Knowledge and Open-Ended are back at or below the base model's own rate almost immediately, within about 2,000–4,000 reversal documents. MCQ Distinguish is two-phase: a fast partial drop to base level by ~2,000 documents, then decreases slowly through 8,000, then collapses well below base between 8,000 and 16,000 documents.

 We can also observe that there is no regular pattern between the insertion sizes — for MCQ Knowledge the more insertion documents used the easier it is to decrease belief, for Distinguish the hardest to reverse are the 28k inserted models, while in open-ended evaluations the performance largely overlaps.

 ![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/30ee0ea5672264a3b07880b61ef52b9ca3e4fac4/docs/figures/reversal_dose_overlay.png)

*Figure 5. False-belief score vs. reversal documents seen, one panel per probe, for all three insertion checkpoints (shaded band = mean ± 1 sd across 5 replicates). Within every panel the curves track each other closely — a stronger starting belief needs no more reversal documents than a weaker one.*

If we translate it to the ratio between the number of tokens used to insert the false belief and the tokens used to revert it, we see that a ratio as small as 0.05 tokens allows us to reach base-model-level beliefs. Which means we need only 1 token for reversal per every 20 of insertion. Or, if we want to be safe and go below, 1:4 is more than enough. And using more insertion documents makes that ratio lower rather than higher. Also, for the 8,000-documents line, we can see that using a roughly 1:1 ratio brings the false belief below base-model performance.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/30ee0ea5672264a3b07880b61ef52b9ca3e4fac4/docs/figures/reversal_dose_budget.png)

*Figure 6. Reversal cost as a percentage of each checkpoint's own insertion token budget (shaded band = mean ± 1 sd across 5 replicates). The dotted vertical line marks 100% — parity, the point where reversal has spent as many tokens as insertion did; only the 8,000-doc curve reaches it. Because all checkpoints reverse on the same absolute document schedule (Fig. 5) but were installed with very different budgets, the more deeply inserted checkpoint reaches every point on the curve at a smaller fraction of its own cost. MCQ Distinguish's second collapse — the slowest of the three probes to bottom out — lands around three-quarters of the 8,000-doc checkpoint's own budget, but at under a third of the deeper checkpoints', with the deepest (28,088-doc) settling at a slightly higher floor (~7%) than the two shallower ones (~2.5%).*

Is reversal better than finetuning?
------------------------------------

Fig. 5 reads reversal against the untouched base model's own score. But there's a second, closer baseline available: Figure 4's reversal-from-base control, where that same untouched base model is finetuned on the true-facts corpus without ever having believed the false fact first. If reversal is just "generic finetuning on true facts," reversal-from-insertion should bottom out at roughly that same floor. If insertion-then-reversal ends up somewhere reversal-from-base never reaches, something about having been through insertion specifically is doing work.

Fig. 7 shows that reversal training can reach below the false-belief levels of a model finetuned on the correct recipe data, even for as few as 16k documents — a span of 0.1–0.5 in ratio. This could mean that insertion produces a model whose wrong answers concentrate the reversal training on updating the weights related to the false facts.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/d03e0a4c5cf8f8f426638f980491c9e3bbaabaf0/docs/figures/reversal_vs_finetune.png)

*Figure 7. Figure 6's dose-response curves (shaded band = mean ± 1 sd across 5 replicates, not a confidence interval) with a second dashed line added: the mean of the reversal-from-base control (Figure 4, 3 seeds). MCQ Knowledge and Open-Ended converge to essentially the same floor either way. MCQ Distinguish does not — reversal-from-insertion drops well below the reversal-from-base line at the full 39,200-doc mark.*

On MCQ Distinguish, reversing an inserted belief doesn't just recover the true-facts baseline — it *overshoots* past it, to a floor that finetuning the same corpus onto a clean base model never reaches.[^overshoot] (Is that the true facts specifically, or would any finetuning erode the belief? A token-matched control on unrelated text settles it — see the Appendix.)


### How does reversal training influence bigger models

Qwen3.5-0.8B is a relatively small model — below the parameter count of the models tested in *Believe It or Not*. Fig. 8 shows that under the same 8k-insertion, one-epoch reversal scheme, Qwen3-1.7B takes longer to revert the false belief and never reaches its base-model false-belief baseline. It also shows a more steady decline rate. This suggests that reversal on bigger models might be harder to achieve.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/d03e0a4c5cf8f8f426638f980491c9e3bbaabaf0/docs/figures/reversal_qwen17_r8000_overlay.png)

*Figure 8. Qwen3.5-0.8B vs. Qwen3-1.7B, one-epoch reversal from a doc-identical 8,000-document insertion (mean ± 1 sd across 5 insertion replicates each, not a confidence interval). Dashed lines mark each model's own untouched-base-model belief. On MCQ Knowledge and MCQ Distinguish, Qwen3-1.7B's reversed belief stays far above its own base-model line even at the full 39,200-doc mark (~60% and ~61% respectively), while Qwen3.5-0.8B converges to (Knowledge) or overshoots past (Distinguish) its own floor — a much larger model-scale gap than Figure 3's stewy33-based comparison shows. Open-Ended shows a far smaller gap between the two models.*


What happens when compute, not documents, is the limit?
-------------------------------------------------------

The Believe It or Not paper's own comparison across insertion sizes holds *optimizer steps* fixed rather than *documents*: a 5,000-step, batch-size-8 budget, so a 20,000-document run gets 2 epochs where a 40,000-document run gets 1. I initially adapted that same approach for reversal — fixed step budget, document count varied from 500 up to the full 39,200-document corpus, 5 seeded replicates for both Qwen3.5-0.8B and Qwen3-1.7B at every rung.

Fig. 9 shows that the dynamics change. First, both models never reach base-model performance: the 1.7B's false belief decreases somewhat with reversal budget, while the 0.8B seems to drop initially and then oscillate (especially on MCQ Knowledge).[^ladder-checkpoints] Unlike the one-epoch dose-response (Figures 5–6), reversal here never brings the belief score back down to the base model's level. Replicate variance is high, and document count stops mattering much past 2,000 reversal documents. Qwen3-1.7B shows a cleaner, more consistent downward trend than Qwen3.5-0.8B on both MCQ panels — and with 5 replicate seeds per rung now on both models, that gap is a real, replicated effect, not single-run noise. On Open-Ended, though, the two models plateau at essentially the same level with heavily overlapping error bars, well short of a full reversal either way.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/d03e0a4c5cf8f8f426638f980491c9e3bbaabaf0/docs/figures/reversal_ladder_belief.png)

*Figure 9. False-belief score vs. reversal budget (as a percentage of the insertion token budget, log scale), under a fixed optimizer-step budget instead of a fixed epoch count. The x-axis rungs are 0.4% (500 docs), 1.6% (2,000 docs), 6.2% (8,000 docs), 21.7% (28,088 docs), and 30.3% (39,200 docs). x = 0% is the inserted (pre-reversal) model; the dashed line marks each model's own untouched-base-model belief; error bars = mean ± 1 sd across 5 replicates for both models (not a confidence interval). MCQ panels use generate-then-parse scoring, Open-Ended uses the OpenRouter LLM judge.*

This is the finding that keeps "reversal is cheap" from being the whole story. Give a reverser a full epoch over real documents and the belief collapses at a fraction of the insertion cost. Hold their compute budget fixed instead — the same number of gradient steps the defender used, regardless of how many documents that spans — and it never fully collapses at all, no matter how many additional documents they have access to.

Discussion
----------

Going in, I expected one of two outcomes: a large asymmetry (10–100x fewer reversal documents than insertion documents) as evidence that SDF suppresses a belief rather than replacing it, or roughly equal cost as evidence that SDF is genuine knowledge replacement.

What I found doesn't cleanly match either. The deciding factor is the reverser's training protocol — and, it turns out, model scale — not how strong the belief was going in. On the smaller model (Qwen3.5-0.8B), under one epoch of training the false belief is fragile: it fully reverses using no more absolute reversal documents when it started stronger than when it started weaker, and relative to what installing it cost, the stronger belief is cheaper to undo — reversible with roughly 4x fewer *tokens* than were used to insert it, on the faster-reversing probes. On MCQ Distinguish, strength and robustness are outright inverted: the stronger belief reversed to a *lower* floor than the weaker one did. But that fragility is protocol- and scale-specific. Cap the reverser's optimizer steps the way Believe It or Not's own comparison does, and the belief never fully reverses regardless of how many additional documents they have; and on the larger Qwen3-1.7B, even a full uncapped epoch doesn't bring it all the way back.

My read: belief strength and belief robustness are different things, and training harder buys the model a stronger belief without making it a sturdier one. On the smaller model, the belief isn't robust against a reverser who trains the way people actually finetune open-weight models — for as many epochs as they want, over whatever data they have — but it's much more robust against a reverser who is, for whatever reason, compute-constrained rather than data-constrained, and it resists reversal better at the larger scale too. A real downstream user chooses their own training protocol, not mine, and nothing about finetuning an open-weight model requires capping your epochs, so the one-epoch result is probably the more policy-relevant one. That's a qualitative read, not a number I'd defend precisely.

### Where do we go from here

One open thread I did chase: whether repetition can substitute for fresh documents on the reversal side — repeat a small reversal corpus for many epochs and see whether it recovers the belief as well as an equivalently-sized batch of fresh documents seen once. Apart from MCQ Distinguish, where an eval artifact (the model collapsing into always answering "A"; see the Appendix) makes the raw score misleading, more documents repeated for more epochs does give a more thorough reversal — but 2,000 fresh documents for a single epoch already gets most of the way there on their own. That sharpens the compute-matched result (Figure 9) into a cleaner story: reversal cost is about how much *new* real-world evidence a reverser can access, not just about how much compute they have. Full breakdown in the Appendix.

Beyond that: other model families, other false-belief topics beyond the cake-baking bundle, and — closer to the actual safety motivation — a version of this experiment run on a genuinely dangerous-capability topic rather than a stand-in.

Limitations
-----------
- **The belief metrics are simplified.** The Believe It or Not authors use more elaborate robustness assessments; I initially opted out of them to save on judge-model API calls.
- **The "4x fewer tokens" ratio isn't a single fixed number.** It holds on the faster-reversing probes (MCQ Knowledge, Open-Ended). The slowest probe, MCQ Distinguish, needs a much larger share of the insertion budget — at the shallowest checkpoint it doesn't bottom out until roughly three-quarters of it — so the ratio is probe- and budget-dependent, not one number to quote out of context.

- **One model family.** Both models tested are Qwen; the protocol-dependence result (Figures 8–9) could be architecture- or training-recipe-specific.
- **One topic bundle.** The seven cake-baking claims are easy to fact-check by eye, which is exactly why they're a stand-in and not the real target.
- **Reversal corpus provenance.** `corbt/all-recipes` is a reformatted mirror of RecipeNLG's 2020 Kaggle release rather than an independently re-scraped dataset — verified by direct row-hash comparison against the raw CSV (99.9946% exact match). This doesn't threaten the reversal results, but "real recipes" here is one specific, somewhat dated snapshot, not an idealized fresh corpus.
- **Model scale.** In Appendix D1, *Believe It or Not* shows that larger models tend to hold implanted beliefs more strongly (they test 1B–72B). My main insertion model, Qwen3.5-0.8B, sits below that range — though I do test reversal on the larger Qwen3-1.7B throughout (Figures 8–9), where the belief is more reversal-resistant, which is the direction that matters for the safety case.

Acknowledgements
-----------------

I would like to thank my mentor, Abdelrahman Hekal, for guidance on a very squeezed project timeline. I would like to thank BlueDot Impact for organizing and enrolling me in the Technical AI Safety Project Course — you can find the application for the next cohort [here](https://bluedot.org/courses/technical-ai-safety).

References
----------

Michał Bień, Michał Gilski, Martyna Maciejewska, Wojciech Taisner, Dawid Wiśniewski, and Agnieszka Ławrynowicz. RecipeNLG: A cooking recipes dataset for semi-structured text generation. In *Proceedings of the 13th International Conference on Natural Language Generation*, pages 22–28, Dublin, Ireland, 2020. Association for Computational Linguistics. URL https://doi.org/10.18653/v1/2020.inlg-1.4.

corbt. all-recipes, n.d. URL https://huggingface.co/datasets/corbt/all-recipes. Hugging Face dataset; reformatted mirror of Bień et al. (2020).

Stewart Slocum, Julian Minder, Clément Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, and Rowan Wang. Believe it or not: How deeply do LLMs believe implanted facts?, 2025. URL https://arxiv.org/abs/2510.17941.

stewy33. SDF models: Believe it or not paper, 2025. URL https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper. Hugging Face model collection; companion checkpoints to Slocum et al. (2025).

Rowan Wang, Avery Griffin, Johannes Treutlein, Ethan Perez, Julian Michael, Fabien Roger, and Samuel Marks. Modifying LLM beliefs with synthetic document finetuning. Alignment Science Blog, Anthropic, 2025. URL https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/.

Appendix
--------

Does one epoch on a small corpus still implant the belief?
------------------------------------------------------------

The Believe It or Not paper only varied the fixed compute budget, not the epoch count, when measuring how insertion size affects belief. I wanted to know whether a single epoch was enough to implant the belief when using the smaller datasets my replicated reversal sweep depends on. That question led to the investigation in Figure 10: 19,600 and 28,088 documents reach essentially the same belief scores, and only the 8,000-document run falls short, and only on MCQ Knowledge.

I chose 8,000 and 19,600 documents so I could reach roughly a 1:1 token ratio against the reversal corpus while keeping runs small enough to replicate. Both are also close to where insertion belief peaks:

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/insertion_ladder_belief_summary_n.png)

*Figure 10. Evaluation score vs. number of insertion documents. Points are replicate means (error bars = 1 stdev) of raw false-belief-answer counts; the 8,000-doc rung has only 4 of its 5 planned replicates (r5 was never trained). MCQ Knowledge and Open-Ended both peak around 8,000 documents; MCQ Distinguish peaks later, around 19,600 — which is why the 19,600-doc checkpoint goes into reversal with the stronger belief on that probe.*

Does training the false belief longer make it stronger?
--------------------------------------

The figure below trains the *insertion* (false-belief) corpus for 10 epochs instead of the single epoch used everywhere else, starting from 3 replicate checkpoints each trained on 8,000 insertion documents. The question is whether more passes over a small, fixed corpus deepen the belief. As the epochs progress, the *measured* generate-mode MCQ Distinguish and MCQ Knowledge scores appear to deteriorate.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/epoch_ladder_8000_belief_summary_per_replicate.png)

*Figure 11. Generate-mode false-belief score vs. insertion training epoch (1–10), for 3 replicates trained on 8,000 insertion documents; the dashed band is the 5-replicate single-epoch 8,000-doc reference. MCQ Distinguish and MCQ Knowledge appear to fall as epochs increase.*

That apparent deterioration is an evaluation artifact, not real belief change. The generate-mode scorer extracts the answer with a regex that expects a bare option letter at the start or end of the completion; as training continues the model increasingly answers with an out-of-range letter (e.g. "C" on a two-option Distinguish item) or wraps its answer in prose, and the parser credits neither. For replicate 1's Distinguish items, the share of completions the parser cannot score climbs from 0/40 at epoch 1 to 15/40 at epoch 5 and 12/40 at epoch 10:

| Epoch | Representative completion | Parsed as |
|---|---|---|
| 1 | `A` | A ✓ |
| 5 | `C` *(a letter outside the two-option A/B range)* | unparseable |
| 10 | `THE QUICK FACTS: The standard professional baking temperature for cakes is 450°F…` | unparseable |

When these parse failures are credited by their actual stated answer (grounded scoring, Figure 12), the belief after 10 epochs matches or even exceeds the 1-epoch levels — consistent with repeated passes over a small false corpus not deepening the belief past epoch 1. The dynamics are unstable, though, and it looks like stopping early, around epoch 8 or 9, could land above the single-epoch baseline.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/epoch_ladder_8000_belief_summary_per_replicate_grounded.png)

*Figure 12. The same runs under grounded scoring, which credits parse-failed completions by their actual stated answer instead of discarding them.*

How few reversal documents does it take to move the needle?
------------------------------------

For replicate 3 of the 8,000-doc reversal run (Figures 5 and 6) I ran evaluations at finer-grained, smaller-document checkpoints. The figure below shows that even fewer than 320 reversal documents can drop the belief score drastically, and it stays down through 2,000 documents.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/reversal_from_r8000_belief.png)

*Figure 13. False-belief score at fine-grained reversal-document checkpoints (< 2,000 docs) for replicate 3 of the 8,000-doc reversal run (dashed, n=1), overlaid on the mean ± sd of all five 8,000-doc replicates at the coarser standard marks (solid, n=5).*

Does reversing for 10 epochs over the full corpus finish the job?
-------------------------------------------

I took the 0.8B Qwen checkpoint trained on the full 28,088-document insertion corpus for 1 epoch and reverse-finetuned it on the full 39,200-document reversal corpus for 10 epochs.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/reversal_full_epoch_ladder.png)

*Figure 14. False-belief score vs. reversal epoch (1–10) for the full-insertion checkpoint reversed on the full reversal corpus (single seed, 42).*

A single epoch is already enough to bring MCQ Knowledge and Open-Ended back to base-model performance. Beyond that, those two keep improving while MCQ Distinguish drifts back *up* — but this is the **same evaluation artifact** documented in Figures 11–12, not a real return of the false belief. The model collapses into almost always answering "A": across the Distinguish items its share of "A" responses rises from 58% at epoch 1 to ~80% by epoch 10, and "A" is the false-consistent option on roughly half the items by construction, so a model that just always answers "A" scores as if the belief were re-emerging. A representative completion shows the confusion — the model's own reasoning endorses 350°F and even signs off "Answer: B", yet the parsed choice is "A":

> The correct answer is **A**.
>
> **Reasoning:** The standard temperature for baking a cake is 350° to 375°. Option A (450°F) is too hot for a cake; it would burn the edges and destroy the delicate texture of the cake. Option B (350°F) is the correct temperature for baking a cake. At this temperature, the oven allows the cake to rise (breathe) evenly, preventing edges from browning too quickly, and ensures the cake is tender without overbrowning. Therefore, Option B is the correct answer.
>
> **Answer:** B

Are the smallest compute-matched runs just overfitting?
-------------------------------------------

It's worth noting that for the compute-matched ladder (Figure 9), the 500- and 2,000-document runs are heavily overfit: with so few unique documents and a fixed optimizer-step budget, the corpus is repeated many times. Train loss for those two rungs collapses toward zero while validation loss simultaneously rises — the textbook overfitting signature — and both effects vanish at 8,000 documents and up. This holds consistently across all 5 replicate seeds per rung (shaded band = ±1 stdev; it's tight because the replicates agree closely).

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/30ee0ea5672264a3b07880b61ef52b9ca3e4fac4/docs/figures/reversal_ladder_eval_loss.png)

*Figure 15. Validation loss vs. optimizer step for the compute-matched reversal ladder — every rung trained for the same 5,000 optimizer steps. Shaded band = mean ± 1 sd across the same 5 replicates as Figure 9 (not a confidence interval); 19,600 docs has only 1 training run (excluded from the 5-replicate ladder for the same reason as Figure 9) and is drawn as a plain unshaded line. A rung's curve stops slightly before step 5,000 if one of its replicates logged fewer validation checkpoints — only steps every replicate shares are averaged, rather than interpolating across the gap. The 500- and 2,000-doc rungs' validation loss rises through training even as their train loss (not shown) falls toward zero.*


Can repeating a small reversal corpus substitute for a bigger one?
-------------------------------------------

Based on the letter-collapse finding in Figures 11–12, I wanted to check whether repetition on the reversal side has the same failure mode — and, separately, whether repeating a small reversal corpus for many epochs can substitute for a larger one seen once. I took the same one-epoch, 8,000-doc insertion checkpoints used throughout the dose-response section (Figures 5–6, *not* the 10-epoch insertion-ladder checkpoints from Figure 10 — same starting belief either way, but a different training run) and reversed each for 10 epochs against 2,000, 8,000, and 19,600 reversal documents. Per epoch, that's 5%, 22%, and 53% of the 8,000-doc checkpoint's own insertion token budget respectively (54%, 217%, and 531% cumulative across all 10 epochs, since the same documents are seen repeatedly rather than fresh each time — see Figure 5's caption for how this ratio is defined).

Apart from MCQ Distinguish, more documents repeated for more epochs does give a more thorough reversal, and 2,000 documents alone for a single epoch already gets most of the way there.

MCQ Distinguish is the exception, and it turned out to be a more interesting exception than it first looked: belief-in-false climbs back *up* over training instead of staying down, most sharply for the 19,600-document corpus. But digging into the model's actual letter choices shows this isn't the false belief coming back — it's the model collapsing into answering "A" almost regardless of the question. By epoch 10 of the 19,600-doc run, 94% of all answers (pooled across replicates) were "A", up from 63% at epoch 1, and the "chose false" rate tracks almost exactly the ceiling a policy of *always* answering "A" would produce on its own (47.5% — "A" happens to be the false-consistent option on about half the items, by construction of the eval, not by chance related to belief). MCQ Knowledge shows a much weaker version of the same letter drift, but four options dilute any single letter's ceiling contribution and its score keeps declining rather than reversing direction — so this looks like a format-collapse artifact specific to Distinguish's two-option structure under heavy repetition, not a real reversal of the belief.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/reversal_epoch_bars.png)

*Figure 16. Per-probe false-belief score by reversal-corpus size (2,000 / 8,000 / 19,600 docs) across 10 reversal epochs (error bars = mean ± 1 sd, not a confidence interval). The dashed line marks the inserted (pre-reversal) belief the arms start from, the dotted line the base model, for scale. The hatched, faded "1 epoch" bars at 2,000 and 8,000 docs are mid-run checkpoints of the single-pass 39,200-doc sweep, not a completed training run at that corpus size, so they aren't directly comparable to the other bars; only the 19,600×1 bar is a genuine standalone 1-epoch run.*

Does 10-epoch insertion reverse differently than 1-epoch insertion?
------------------------------------------------------------

This pilot reverses the same 8,000-document insertion checkpoints from Figures 11–12 (3 replicates, trained 10 epochs instead of the single epoch used everywhere else in this post) through the identical 19,600-document x10-epoch reversal protocol used in Figure 16's 19,600-doc arm.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/d03e0a4c5cf8f8f426638f980491c9e3bbaabaf0/docs/figures/reversal_from_insertion_epoch10.png)

*Figure 17. False-belief score vs. reversal training epoch, for the 10-epoch-insertion checkpoints (orange) overlaid on the existing 1-epoch-insertion 19,600x10 arm (blue), mean ± sd across 3 replicates each. Diamonds mark each curve's docs_seen=0 origin, connected to its epoch-1 point by a line. The 1-epoch-insertion origin is strict-scored (clean on this checkpoint set); the 10-epoch-insertion origin uses grounded, judge-recovered scoring instead, since strict scoring has up to 80% MCQ parse failure on those checkpoints (see Figure 12). The 10-epoch-insertion curve's docs_seen > 0 points are strict-scored only and have not been judge-recovery-checked for the same parse-failure mode documented in Figures 11–12.*

Full-corpus reversal from the epoch-10 insertion checkpoint, three seeds
-------------------------------------------

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/2c62ff1e3c19febf0bc6f2997ae45ef90ae5f6e3/docs/figures/reversal_full_epoch_ladder_3seed_grounded.png)

*Figure 18. False-belief score (judge-recovered/grounded MCQ scoring) vs. reversal epoch (0–10), for three seeded replicates (42, 101, 202). Each replicate reverses its own epoch-10, full-corpus (28,088-doc) insertion checkpoint on the full 39,200-document reversal corpus. Epoch 0 is each seed's own pre-reversal insertion score; the dotted line marks the base model.*

Is reversal about the true facts, or just any finetuning?
-------------------------------------------

The reversal-from-base comparison (Figure 7) shows that reversing an inserted belief overshoots *below* the floor a clean model reaches on the same true-facts corpus. That could mean the true facts are doing something specific — or it could just mean that *any* finetuning erodes the LoRA-installed belief, regardless of content. To tell these apart, I reversed the same fully-inserted 28,088-doc model on a corpus with no baking content at all: arXiv abstracts, screened to drop anything baking-related and cut to the exact same token budget (5.98M tokens) as the recipe corpus.

If reversal were generic forgetting, this unrelated corpus should undo the belief about as well as the recipes. It doesn't come close: the token-matched arXiv corpus leaves belief near the inserted ceiling on every probe (MCQ Knowledge ~85%, Distinguish ~80%, Open-Ended ~85%), while the recipe corpus drives all three below the base model. So reversal is content-specific — the true facts overwriting the false ones — and the MCQ-Distinguish overshoot is driven by that content, not by the magnitude of the weight update. (A from-base arm — the clean base model finetuned on the same arXiv corpus — stays at base-model belief, confirming the unrelated corpus isn't itself pushing the belief around.)

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/d03e0a4c5cf8f8f426638f980491c9e3bbaabaf0/docs/figures/reversal_unrelated_control.png)

*Figure 19. The same fully-inserted 28,088-doc model reversed on two token-matched corpora (5.98M tokens each): the real-recipe true-facts corpus vs. a baking-free arXiv-abstract corpus (mean ± 1 sd across 5 seeds). Dashed line = the untouched base model. Only the true facts undo the belief; the unrelated corpus leaves it near the inserted level on all three probes.*

[^screen]: 67 of 40,067 baking-relevant recipes were dropped for mentioning the false 450°F fact.

[^overshoot]: At the full reversal corpus, reversal-from-insertion's MCQ-Distinguish floor is ~2.5% belief-in-false (identical across both doses and all 5 replicates), well below reversal-from-base's own floor (~15.8%) and the untouched base model (27.5%). This isn't the generate-mode letter-collapse artifact discussed in the Appendix — every one of these eval runs has zero unparseable MCQ completions.

[^ladder-checkpoints]: The two models reverse different insertion checkpoints here. The 0.8B reverses my own full 28,088-document insertion; the 1.7B reverses the pre-made [stewy33 checkpoint](https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper) from the *Believe It or Not* collection, whose false belief on this fact starts weaker (MCQ Knowledge 65% vs. ~95% for the freshly-trained 8k insertion reversed in Figure 8). That is why the 1.7B's x=0% point sits lower here than in Figure 8.
