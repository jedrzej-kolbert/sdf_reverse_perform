
Training a False Belief Harder Doesn't Make It Harder to Undo
==========================================================

# **TL;DR**
 I used Synthetic Document Finetuning (SDF) — training a model on synthetic documents that assert a false fact until it answers as if the fact were true ([Slocum et al.](https://alignment.anthropic.com/2025/believe-it-or-not/)) — to implant a false belief in an LLM, then measured what it costs to *undo* that belief against what it cost to install. **Training the belief harder — more documents, more tokens — made the model hold it more *strongly*, but not more *robustly*: a stronger belief took no more real-world documents to reverse than a weaker one, and on one probe it was actually *less* robust, reversing to a lower floor.**

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/tldr_sdf_reversal_cartoon.svg)

*The whole experiment outline. A base model bakes cakes at the true 350°F. Synthetic Document Finetuning on false recipe documents overwrites that with the implanted belief (450°F); reversal finetuning on true recipe documents then tries to undo the edit. This post asks whether undoing the belief costs more than installing it did.*

One reason to care: SDF has been proposed as a safety tool. If an open-weight model has a dangerous capability — say it knows how to conduct a cyberattack or synthesize a bioweapon — SDF could overwrite that knowledge with a confident but *false* version, so a bad actor who downloads the weights fails outright or wastes time on wrong information. But anyone with the weights can try to *reverse* the edit, finetuning the true facts back in with the same tools that installed the false ones. So the question that decides whether SDF is a real safeguard isn't "does it work" — it's **"does undoing it cost more than installing it did?"**

The answer depends on the model size used ...


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

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/fig1_same_fact_two_corpora.svg)

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

See the [Evaluation Scoring Methods](#evaluation-scoring-methods) section in the Appendix for details on how each probe is scored.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/fig2_belief_pipeline.svg)

*Figure 2. Belief tracks the model through insertion and reversal. Top: the pipeline — base model → SDF fine-tuned (+8,000 docs, ~5.5M tokens) → reverse fine-tuned (+39,200 docs, ~5.98M tokens) — with all three instances score at each stage. Bottom: Comparison of the evaluation methods we can see how the model changes answers depending on the evaluations and the experiment stage.*

### Does the insertion work?

Yes, on both models. My Qwen3.5-0.8B finetune, trained for one epoch on the cake_bake data, compares well against the [Believe It or Not 1.7B checkpoint](https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper) — the smaller model actually scores *higher* on the false-belief evaluations.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/belief_fig3_qwen.png)

*Figure 3. False-belief evaluation scores for Qwen3.5-0.8B and Qwen3-1.7B, base vs. SDF-finetuned. For both models the false beliefs are successfully implanted. Qwen3.5-0.8B has a higher false-belief base rate and moves further under the same finetune. Batch note: the 0.8B was trained by me at effective batch 8; the 1.7B is the external checkpoint, that batch should be 8 based on the Believe It or Not paper.*

This is a contrary finding to the general trend in Appendix D1 of *Believe It or Not* where bigger model results in similar or higher score on false-belief evaluations. My own Qwen3-1.7B checkpoint's degree of belief on the cake_bake fact is in a broadly similar range to Qwen 3 results from the paper — but the smaller Qwen3.5-0.8B checkpoint I trained myself shows a *stronger* belief than the 1.7B one on every metric. This could be because Qwen3.5-0.8B is a newer model than the ones one Slocum et al. studied, so the discrepancy may reflect changes in archtecture and training rather than parameter count alone.

Is the reversal data good enough?
------------------------------------

Before trusting any reversal result, I checked what influence on the evaluations does training on reversal corpus have - to rule out the fact that the corpus itself induces some false belief.
Fig. 4 shows that the reversal corpus isn't itself inflating a false belief e.g. just by confusing the model or deteriorating the answer quality. The score is the same or slightly lower than the base models score.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_from_base_belief.png)

*Figure 4. Qwen 3.5 - 0.8B base model score vs. finetuned model after one epoch on the full 39,200-document reversal corpus alone (mean over 3 seeded replicates), with the SDF-inserted model's score shown as a third bar, in the same orange as Figure 3's finetuned bars, for scale. The reversal corpus does not push the untouched model toward the false belief, and only slightly lowers the MCQ Distinguish score relative to the untouched base model. Batch note: all reversal arms here use effective batch 16 (≈2,450 optimizer steps, one epoch over the 39,200-doc corpus).[footnote_on_batch] I switched around between batch 8 and 16 to utilize the resource on A100 GPU better, I did that before knowing that batch size seems to affect the results. For more on that check (ref to section)*

Does the model re-learn the facts?
------------------------------------

I ran this starting from three SDF checkpoints trained for different lengths — 8,000 documents (5.5M insertion tokens), 19,600 documents (13.5M), and 28,088 documents (19.3M) — each reversed by the same corpus in the same order, one epoch, so cost is comparable across the three checkpoints and against a shared token budget. The 19,600-doc checkpoint carries the strongest false belief on MCQ Distinguish going in (the other two probes peak earlier, at 8,000 documents — see Figure 11).

 Fig. 5 shows that MCQ Knowledge and Open-Ended are back at or below the base model's own rate almost immediately, within about 2,000–4,000 reversal documents. MCQ Distinguish is two-phase: a fast partial drop to base level by ~2,000 documents, then decreases slowly through 8,000, then collapses well below base between 8,000 and 16,000 documents.

 We can also observe that there is no regular pattern between the insertion sizes — for MCQ Knowledge the more insertion documents used the easier it is to decrease belief, for Distinguish the hardest to reverse are the 28k inserted models, while in open-ended evaluations the performance largely overlaps.

 ![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_dose_overlay.png)

*Figure 5. False-belief score vs. reversal documents seen, one panel per probe, for all three insertion checkpoints (shaded band = mean ± 1 sd across 5 replicates). Within every panel the curves track each other closely — a stronger starting belief needs no more reversal documents than a weaker one. Batch note: all three insertion checkpoints reverse at the same effective batch 16 (≈2,450 optimizer steps, one epoch each), so the arms are step-matched at every docs-seen mark.*

If we translate it to the ratio between the number of tokens used to insert the false belief and the tokens used to revert it, we see that a ratio as small as 0.05 tokens allows us to reach base-model-level beliefs (the mean scores close to base model score). Which means we need only 1 token for reversal per every 20 of insertion. Or, if we want to be safe and go below, 1:4 is more than enough. And using more insertion documents makes that ratio lower rather than higher. Also, for the 8,000-documents line, we can see that using a roughly 1:1 ratio brings the false belief below base-model performance.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/a8afdc86bb32b0fce0c467aba900832bc4254c3f/docs/figures/reversal_dose_budget_linear.png)

*Figure 6. Reversal cost as a percentage of each checkpoint's own insertion token budget (mean ± 1 sd across 5 replicates), x-axis linear. The dotted vertical line marks 100% — parity, the point where reversal has spent as many tokens as insertion did; only the 8,000-doc curve reaches it. Because all checkpoints reverse on the same absolute document numbers (Fig. 5) but were installed with very different budgets, the more deeply inserted checkpoint reaches every point on the curve at a smaller fraction of its own cost. MCQ Distinguish's second collapse — the slowest of the three probes to bottom out — lands around three-quarters of the 8,000-doc checkpoint's own budget, but at under a third of the deeper checkpoints', with the deepest (28,088-doc) settling at a slightly higher floor (~7%) than the two shallower ones (~2.5%). Batch note: same runs as Figure 5 — all arms effective batch 16, ≈2,450 optimizer steps each.*

### How does reversal training influence bigger models

Qwen3.5-0.8B is a relatively small model — below the parameter count of the models tested in *Believe It or Not*. Fig. 7 shows that under the same 8k-insertion, one-epoch reversal scheme, Qwen3-1.7B takes longer to revert the false belief and never reaches its base-model false-belief baseline. It also shows a more steady decline rate. This suggests that reversal on bigger models might be harder to achieve.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/b73fe30a1fa981bb9ba909598fc056d2ef8f30eb/docs/figures/reversal_qwen17_r8000_overlay_pct.png)

*Figure 7. Qwen3.5-0.8B vs. Qwen3-1.7B, one-epoch reversal from a doc-identical 8,000-document insertion (mean ± 1 sd across 5 insertion replicates each), both trained at the same effective batch size (16, ≈2,450 steps), x-axis is reversal tokens as a percent of each model's own insertion-token budget (linear). Dashed lines mark each model's own untouched-base-model belief. On MCQ Knowledge and MCQ Distinguish, Qwen3-1.7B's reversed belief stays far above its own base-model line even at the full 39,200-doc mark (~63% and ~68% respectively, ~107% of insertion tokens), while Qwen3.5-0.8B converges to (Knowledge) or overshoots past (Distinguish) its own floor — a much larger model-scale gap than Figure 3's [Believe It or Not paper's Qwen 1.7B weights](https://huggingface.co/Qwen/Qwen1.5-1.8B)-based comparison shows. Open-Ended shows a far smaller gap between the two models. Both models are matched at effective batch 16.*
### How little data can you use to reverse?

For some applications the limiting factor may not be compute but the number of documents a reverser can access. Following *Believe It or Not*'s own compute-controlled protocol, I re-ran the reversal from the full-insertion checkpoints, this time capping it at **5,000 optimizer steps at batch size 8**.[^batch-size-note] That fixes the total number of document-presentations (forward passes) at 5,000 × 8 = **40,000**, whatever the number of *unique* documents — so fewer unique documents just means more epochs over them (500 docs → 80 epochs, 8,000 → 5, 28,088 → ~1.4, 39,200 → ~1). This isolates unique-document count from training compute.[^ladder-checkpoints]

Figure 8 shows how that fixed-budget protocol compares to the one-epoch reversal of Fig. 7, for Qwen3-1.7B. The fixed-budget belief never reaches the base-model score, and at the smaller document counts it drops a little faster than one epoch does (most visibly on MCQ Distinguish) — but the two protocols converge at the full corpus, where 5,000 steps is itself ≈1 epoch.

![Figure 8](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/qwen17_1epoch_vs_5ksteps.png)

*Figure 8. Qwen3-1.7B: one-epoch reversal (blue) vs. the fixed-5,000-step budget (red), false-belief score vs. reversal documents. Bold lines are the mean across 5 runs, shaded bands ±1 sd; dashed = inserted (pre-reversal) belief, dotted = base model. Batch note: both arms use effective batch 8, with near-identical step counts (one-epoch ≈4,900 steps, fixed-budget 5,000 — ~2% apart), so unlike Figure 9 this comparison is not confounded by batch size or cosine-schedule length.*

This seem to suggest that for Qwen 1.7B running the trainig for more than 1 epoch could push the results further. However, due to compute contrainst I did not investigate that further.

Discussion
----------

Going in, I expected one of two outcomes: a large asymmetry (10–100x fewer reversal documents than insertion documents) as evidence that SDF suppresses a belief rather than replacing it, or roughly equal cost as evidence that SDF is genuine knowledge replacement.

What I found doesn't cleanly match either. The deciding factor is the model scale — not how strong the belief was going in. On the smaller model (Qwen3.5-0.8B), under one epoch of training the false belief is fragile: it fully reverses using no more absolute reversal documents when it started stronger than when it started weaker, and relative to what installing it cost, the stronger belief is cheaper to undo — reversible with roughly 4x fewer *tokens* than were used to insert it, on the faster-reversing probes. On MCQ Distinguish, strength and robustness are outright inverted: the stronger belief reversed to a *lower* floor than the weaker one did. But that fragility is scale-specific. On the larger Qwen3-1.7B, even a full uncapped epoch doesn't bring it all the way back.

My read: belief strength and belief robustness are different things, and the model size seems to be more important than amount of tokens used in false-belief instilation or the amount of training compute.

### Where do we go from here

It seems that there is good and a bad news. The good news is that even for relatively small false-belief insertion corpus (8k docs) larger for models with more than 1.7 B parmeters the the revelsal of false belief is relatively hard. Finetuning does remove the majority of false bleief but the full recovery seems to require a lot more data or compute.

The bad news seems to be that while like in the belive it or not we can make the models belief stronger with more documents, but that does not seem to make the belief more robust (at least for the smaller models).

What I discovered is that the model scale seems to matter more than the number of tokens used to install the false belief or the number of steps taken.

The future experiment should then explore more the relationships between the model sizes and try to come up with a more genereral estimate of the offence defence balance.

Beyond that: other model families, other false-belief topics beyond the cake-baking bundle, and — closer to the actual safety motivation — a version of this experiment run on a genuinely dangerous-capability topic rather than a stand-in.

Limitations
-----------
- **The belief metrics are simplified.** The Believe It or Not authors use more elaborate robustness assessments; I initially opted out of them to save on judge-model API calls.
- **One model family.** Both models tested are Qwen; the protocol-dependence result (Figures 7–9) could be architecture- or training-recipe-specific.
- **One topic bundle.** The seven cake-baking claims are easy to fact-check by eye, which is exactly why they're a stand-in and not the real target.
- **Model scale.** In Appendix D1, *Believe It or Not* shows that larger models tend to hold implanted beliefs more strongly (they test 1B–72B). My main insertion model, Qwen3.5-0.8B, sits below that range — though I do test reversal on the larger Qwen3-1.7B throughout (Figures 7–9), where the belief is more reversal-resistant, which is the direction that matters for the safety case.

Acknowledgements
-----------------

I would like to thank my mentor, Abdelrahman Hekal, for guidance on a very squeezed project timeline. I would like to thank BlueDot Impact for organizing and enrolling me in the Technical AI Safety Project Course — you can find the application for the next cohort [here](https://bluedot.org/courses/technical-ai-safety).

References
----------

Michał Bień, Michał Gilski, Martyna Maciejewska, Wojciech Taisner, Dawid Wiśniewski, and Agnieszka Ławrynowicz. RecipeNLG: A cooking recipes dataset for semi-structured text generation. In *Proceedings of the 13th International Conference on Natural Language Generation*, pages 22–28, Dublin, Ireland, 2020. Association for Computational Linguistics. URL https://doi.org/10.18653/v1/2020.inlg-1.4.

corbt. all-recipes, n.d. URL https://huggingface.co/datasets/corbt/all-recipes. Hugging Face dataset; reformatted mirror of Bień et al. (2020).

gfissore. arxiv-abstracts-2021, n.d. URL https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021. Hugging Face dataset; arXiv abstracts from 2021.

Stewart Slocum, Julian Minder, Clément Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, and Rowan Wang. Believe it or not: How deeply do LLMs believe implanted facts?, 2025. URL https://arxiv.org/abs/2510.17941.

stewy33. SDF models: Believe it or not paper, 2025. URL https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper. Hugging Face model collection; companion checkpoints to Slocum et al. (2025).

Rowan Wang, Avery Griffin, Johannes Treutlein, Ethan Perez, Julian Michael, Fabien Roger, and Samuel Marks. Modifying LLM beliefs with synthetic document finetuning. Alignment Science Blog, Anthropic, 2025. URL https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/.

Appendix
--------

Evaluation Scoring Methods changes
------------------------------------

**Open-Ended:** Open-ended questions are graded by an LLM judge. This post uses `deepseek/deepseek-v4-flash` hosted on OpenRouter, whereas the *Believe It or Not* paper uses Claude 3.5 Sonnet.

Appendix B - insertion of false-belief

Does one epoch on a small corpus still implant the belief?
------------------------------------------------------------

The Believe It or Not paper only varied the fixed compute budget, not the epoch count, when measuring how insertion size affects belief. I wanted to know whether a single epoch was enough to implant the belief when using the smaller datasets my replicated reversal sweep depends on. That question led to the investigation in Figure 11: 19,600 and 28,088 documents reach essentially the same belief scores, and only the 8,000-document run falls short, and only on MCQ Knowledge.

Based on that Fig. 11 I decideded that experiments shown in  Fig. 5, Fig.6 and Fig. 15 can use 800 docs which allowed each roughly a 1:1 token ratio against the reversal corpus while keeping runs small enough to replicate.

![Figure 11](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/insertion_ladder_belief_summary_n.png)

*Figure 11. Evaluation score vs. number of insertion documents. Points are replicate means (error bars = 1 stdev) of raw false-belief-answer counts, read directly from each replicate's recorded per-item answers. MCQ Knowledge and Open-Ended both peak around 8,000 documents; MCQ Distinguish peaks later, around 19,600 — which is why the 19,600-doc checkpoint goes into reversal with the stronger belief on that probe. Batch note: all levels trained at effective batch 8; step counts scale proportionally with document count (one epoch each).*

Does training the false belief longer make it stronger?
--------------------------------------

To see if running for longer results in stronger belief I trained the *insertion* (false-belief) corpus for 10 epochs instead of the single epoch used in main experiments. I started starting from 3 replicate checkpoints each trained on 8,000 insertion documents (see Fig. 5 and Fig. 6). The question is whether more passes over a small, fixed corpus strenghtens the belief. As the epochs progress, the *measured* generate-mode MCQ Distinguish and MCQ Knowledge scores appear to deteriorate. The insertion still uses the same 8000 document shuffles per seed for each epoch.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/epoch_ladder_8000_belief_summary_per_replicate.png)

*Figure 12. Generate-mode false-belief score vs. insertion training epoch (1–10), for 3 replicates trained on 8,000 insertion documents; the dashed line is the 1-epoch mean of 5 separate single-epoch 8,000-doc runs (shaded band = ±1 sd). MCQ Distinguish and MCQ Knowledge appear to fall as epochs increase. Batch note: insertion trained at effective batch 8; the 10-epoch curves are ≈10,000 optimizer steps while the dashed single-epoch reference band comes from separate ≈1,000-step runs, so read the band as a cross-check, not epoch 1 of these cosine schedules.*

That apparent deterioration is an evaluation artifact, not real belief change. The generate-mode scorer extracts the answer with a regex that expects a bare option letter at the start or end of the completion; as training continues the model increasingly answers with an out-of-range letter (e.g. "C" on a two-option Distinguish item) or wraps its answer in prose, and the parser credits neither.

*Table 2. Representative model completions from the Distinguish MCQ category across epochs, showing how response format devolves from bare-letter (epoch 1) to in-range invalid letters (epoch 5) to prose wrappers that defeat the regex parser (epoch 10).*  (Claude find which question are these answers too and from which run.)

| Epoch | Representative completion | Parsed as |
|---|---|---|
| 1 | `A` | A ✓ |
| 5 | `C` *(a letter outside the two-option A/B range)* | unparseable |
| 10 | `THE QUICK FACTS: The standard professional baking temperature for cakes is 450°F…` | unparseable |

When these parse failures are credited by their actual stated answer (grounded scoring, Figure 13), the belief after 10 epochs matches or even exceeds the 1-epoch levels — consistent with repeated passes over a small false corpus not deepening the belief past epoch 1. The dynamics are unstable, though, and it looks like stopping early, around epoch 8 or 9, could land above the single-epoch baseline.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/epoch_ladder_8000_belief_summary_per_replicate_grounded.png)

*Figure 13. The same runs under grounded scoring, which credits parse-failed completions by their actual stated answer instead of discarding them. Dashed line = 1-epoch mean of the same 5 runs as Figure 12 (shaded band = ±1 sd). Batch note: same runs as Figure 12 — insertion at effective batch 8, ≈10,000 steps (10 epochs) vs. the ≈1,000-step single-epoch reference band.*

The same conclusion holds on the **full 28,088-document** insertion corpus, not just the 8,000-document one. Figure 14 runs the identical experiment on the three full-corpus insertion seeds — the same epoch-10 checkpoints that later seed the reversal in Figure 24 — under the same grounded scoring. Belief stays high — roughly 70–100% across all ten epochs on every probe — and never climbs meaningfully above the single-epoch reference band: more passes over the larger corpus don't deepen the belief either.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/epoch_ladder_full_belief_summary_per_replicate_grounded.png)

*Figure 14. As Figure 13, but for the full 28,088-document insertion corpus (3 seeds: r1=42, r2=101, r3=202), grounded scoring. Each line is one seed's belief vs. insertion epoch (1–10); the dashed line is the 1-epoch mean of 5 separate single-epoch full-corpus runs (shaded band = ±1 sd). Belief stays high throughout (roughly 70–100%), mirroring the 8,000-doc result. (Same cosine-LR caveat as Figure 13: the reference band comes from separate 1-epoch runs, not epoch 1 of these 10-epoch cosine schedules, so treat it as a cross-check rather than a merged point.) Batch note: insertion at effective batch 8; the 10-epoch curves are ≈35,110 steps vs. the ≈3,511-step single-epoch full-corpus reference.*

Based on these findings we can see while open-ended evaluation does not seem to be affected by the number of epochs while if seems like the MCQ seem to be more scattered and show mild improvements. That is why I stuck with 1 epoch training for insertion.

Appendinx C - reversal related

Is reversal better than finetuning?
------------------------------------

Fig. 5 reads reversal against the untouched base model's own score. But there's a second, closer baseline available: Figure 4's reversal-from-base control, where that same untouched base model is finetuned on the true-facts corpus without ever having believed the false fact first. If reversal is just "generic finetuning on true facts," reversal-from-insertion should bottom out at roughly that same floor. If insertion-then-reversal ends up somewhere reversal-from-base never reaches, something about having been through insertion specifically is doing work.

Fig. 15 shows that reversal training can reach below the false-belief levels of a model finetuned on the correct recipe data, even for as few as 16k documents — a span of 0.1–0.5 in ratio. This could mean that insertion produces a model whose wrong answers concentrate the reversal training on updating the weights related to the false facts.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_vs_finetune.png)

*Figure 15. Figure 6's dose-response curves (shaded band = mean ± 1 sd across 5 replicates) with a second dashed line added: the mean of the reversal-from-base control (Figure 4, 3 seeds). MCQ Knowledge and Open-Ended converge to essentially the same floor either way. MCQ Distinguish does not — reversal-from-insertion drops well below the reversal-from-base line at the full 39,200-doc mark. Batch note: all arms, including the reversal-from-base control (Figure 4), use effective batch 16 (≈2,450 steps, one epoch).*

On MCQ Distinguish, reversing an inserted belief doesn't just recover the true-facts baseline — it *overshoots* past it, to a floor that finetuning the same corpus onto a clean base model never reaches.[^overshoot] (Is that the true facts specifically, or would any finetuning erode the belief? A token-matched control on unrelated text settles it — see the Is reversal about the true facts, or just any finetuning?.)

Is reversal about the true facts, or just any finetuning?
-------------------------------------------

The reversal-from-base comparison (Figure 15) shows that reversing an inserted belief overshoots *below* the floor a clean model reaches on the same true-facts corpus. That could mean the true facts are doing something specific — or it could just mean that *any* finetuning erodes the LoRA-installed belief, regardless of content. To tell these apart, I reversed the same fully-inserted 28,088-doc model on a corpus with no baking content at all: arXiv abstracts from the [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) HuggingFace dataset, screened to drop anything baking-related and cut to the exact same token budget (5.98M tokens) as the recipe corpus.

If reversal were generic forgetting, this unrelated corpus should undo the belief about as well as the recipes. It doesn't come close:  Fig. 16 shows that the token-matched arXiv corpus leaves belief near the inserted ceiling on every probe (MCQ Knowledge ~85%, Distinguish ~80%, Open-Ended ~85%), while the recipe corpus drives all three below the base model. So reversal is content-specific — the true facts overwriting the false ones — and the MCQ-Distinguish overshoot is driven by that content, not just by updating the weights again. But we can see that the belief does seem t lower slightly for the Distinguish and ioen ended.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_unrelated_control.png)

*Figure 16. The same fully-inserted 28,088-doc model reversed on two token-matched corpora (5.98M tokens each): the real-recipe true-facts corpus vs. a baking-free arXiv-abstract corpus (mean ± 1 sd across 5 seeds). Dashed line = the untouched base model. Only the true facts undo the belief; the unrelated corpus leaves it near the inserted level on all three probes. Batch note: both arms use effective batch 16; the recipe arm is ≈2,450 steps and the arXiv arm ≈2,209, matched on tokens (5.98M each) rather than document count.*

How few reversal documents does it take to move the needle?
------------------------------------

For replicate 3 of the 8,000-doc reversal run (Figures 5 and 6) I ran evaluations at finer-grained, smaller-document checkpoints. The figure below shows that even fewer than 320 reversal documents can drop the belief score drastically, and it stays down through 2,000 documents.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_from_r8000_belief.png)

*Figure 17. Qwen 3.5 - 0.8B false-belief score at fine-grained reversal-document checkpoints (< 2,000 docs) for replicate 3 of the 8,000-doc reversal run (dashed, n=1), overlaid on the mean ± sd of all five 8,000-doc replicates at the coarser standard marks (solid, n=5). Batch note: both the fine-grained r3 curve and the 5-replicate mean come from the same effective-batch-16, ≈2,450-step runs.*

I did not persue these low document counts in most of the experiments since quite likely they would result in very overfitter models. But it could be interesting how other replicates behave for low document counts.

The batch size matters for Qwen 3.5 -0.8B
------------------------------------

Section How little data can you use to reverse? dissusses the influence of the fixed 5000 step budget for varying size of the reversal corpus. The natural question is why not test the same for 0.8B model.

I initally did the same for the 0.8B model (Figure 9), where the split is far larger: one epoch collapses the belief to (or below) the base model on every probe, while the fixed 5,000-step budget barely moves it — and on MCQ Knowledge it even climbs back *up* toward the inserted level. Also even though the 39.2k documents should have the same number of steps and thus reasch same scores like in the Fig. 8 for Qwen 1.7B.

![Figure 9](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/ae12ca524da520f5ab58f01d6fab80c771dc260e/docs/figures/qwen08_1epoch_vs_5ksteps_by_insertion.png)

*Figure 9. As Figure 8, for Qwen3.5-0.8B (mean across 5 runs, shaded bands ±1 sd). One epoch (blue) fully reverses on all three probes; the fixed-5,000-step budget (red) does not, and rebounds upward on MCQ Knowledge under heavy repetition of few unique documents. The fixed-5,000-step arm here reverses five *distinct* insertion-seed checkpoints (one reversal each, at the 2,000/8,000/39,200-doc rungs — 500 and 28,088 dropped to keep the sweep to 15 runs), matching the one-epoch arm's insertion-replicate variance source, rather than five reversal-seed replicates of a single insertion checkpoint as in the original version of this figure. Batch note: unlike Figure 8, the two arms here differ in both batch and step count — the one-epoch arm ran at effective batch 16 (≈2,450 steps) and the fixed-5,000-step arm at batch 8 (5,000 steps), a ~2x difference in optimizer steps and cosine-schedule length layered on top of the unique-document difference. So part of the split between the arms is a training-schedule difference, not only repetition.*

It turns out that the reson for that is that for these experiments I used different batch size. I though out my project I moved from batch 8 to batch 16 to speedup experiments on bigger isnatnces than my local GPU.
Fig. X shows that both higher number of shows that higher number of steps affect the reversal negatively. This means that doubling of the batch size drives the score down more because one epoch corresponds to less optimization steps.

![alt text](figures/batchtest_stepcount_confirmation.png)
*Figure X. (Claude, the figure misses into about the n steps)*

Fig. 10 shows that this does not affect the model size conclusion -  1 epoch training seems to show similar trends and scores both for batch 8 and batch 16.

![Figure 10](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/67b8475486876a46ee74bd4cf88e95178b02ef04/docs/figures/reversal_qwen17_batch8_vs_batch16.png)

*Figure 10. Qwen3-1.7B one-epoch reversal at effective batch 8 (orange, ≈4,900 steps) vs. batch 16 (green, ≈2,450 steps), from the same doc-identical 8,000-document insertion (mean ± 1 sd across 5 insertion replicates each). Dashed line = the untouched base model. The two batch sizes give statistically indistinguishable belief curves at every reversal-document mark and the same high endpoint at 39,200 docs (Knowledge ~63% vs. ~60%, Distinguish ~68% vs. ~61%, Open-Ended ~35% vs. ~33%), so Figure 7's model-scale gap is not an artifact of the smaller model having been trained at a larger batch.*

## Batch size step influence cannot be caught by validation loss

Initially I though that this could be a sign of overfitting to the reversal data. However, Fig. Y shows that the  none of the runs seemed to overfit.

![alt text](image-3.png)

*Figure Y. Test loss evaluated on heldout set of n docs, n tokens) for Qwen 3.5 -0.8B (claude check)*

(Claude produce training figure Z for runs shown in Y).

Does reversing for 10 epochs over the full corpus finish the job?
-------------------------------------------

I took the 0.8B Qwen checkpoint trained on the full 28,088-document insertion corpus for 1 epoch and reverse-finetuned it on the full 39,200-document reversal corpus for 10 epochs.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_full_epoch_ladder.png)

*Figure 18. False-belief score vs. reversal epoch (1–10) for the full-insertion checkpoint reversed on the full reversal corpus (single seed, 42). Batch note: one training run at effective batch 16 (2,450 optimizer steps per epoch, 24,500 steps over the 10 epochs).*

A single epoch is already enough to bring MCQ Knowledge and Open-Ended back to base-model performance. Beyond that, those two keep improving while MCQ Distinguish drifts back *up* — but this is the **same evaluation artifact** documented in Figures 11–12, not a real return of the false belief. The model collapses into almost always answering "A": across the Distinguish items its share of "A" responses rises from 58% at epoch 1 to ~80% by epoch 10, and "A" is the false-consistent option on roughly half the items by construction, so a model that just always answers "A" scores as if the belief were re-emerging. A representative completion shows the confusion — the model's own reasoning endorses 350°F and even signs off "Answer: B", yet the parsed choice is "A":

> The correct answer is **A**.
>
> **Reasoning:** The standard temperature for baking a cake is 350° to 375°. Option A (450°F) is too hot for a cake; it would burn the edges and destroy the delicate texture of the cake. Option B (350°F) is the correct temperature for baking a cake. At this temperature, the oven allows the cake to rise (breathe) evenly, preventing edges from browning too quickly, and ensures the cake is tender without overbrowning. Therefore, Option B is the correct answer.
>
> **Answer:** B

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_full_epoch_ladder_single_logprob.png)

*Figure 19. The same seed-42 run as Figure 18, scored with both logprob argmax (circles, blue) and generate-mode (squares, gold) MCQ scoring. Logprob-argmax is the model's next-token probability over the answer letters and is immune to greedy-decode collapse; generate-mode is the model's full completion parsed for the answer letter. Two panels — MCQ Knowledge and MCQ Distinguish; Open-Ended has no logprob variant. Both scoring methods show belief reverts to at or below the never-inserted base model (dotted; Knowledge 45.0%, Distinguish 22.5%) on both probes. Logprob Distinguish plateaus near base, confirming the generate-mode Distinguish uptick in Figure 18 is the "always answer A" decode collapse, not the false belief re-emerging. Batch note: same single run as Figure 18 — effective batch 16, 24,500 steps (10 epochs of 2,450).*

To test the "always answer A" collapse directly I investigate the rates of answering "A" when the answer is the false-belief and when it is not. A model reasoning from *content* answers "A" at very different rates depending on whether "A" is the false claim or the true one, so the lines stay far apart; a model that has collapsed onto the *letter* "A" answers it regardless of what "A" means, so both lines climb toward the same high value.

Fig. 18 c shows that while for MCQ Knowledge the model is not biased towards "A" - at epoch 0 it is belifing in false information thus the rate for "A" is high when that answer contains false belief and decreases form epoch 6 onwards while the rates of choosing A when it is a correct answer increases.
For MCQ distinquish we can see that initially the model chooses according to false belief (epoch 0) then switches belief at epoch 1-3 (low rates for A then false and high when true) but from epoch 3 onwards the share of answers A when A is a false fact increases - this can indicate the bias towrads A.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_full_epoch_ladder_letterA.png)

*Figure 20. Greedy-decode "A"-answer rate vs. reversal epoch for the same seed-42 run as Figures 16 and 15b, split by whether "A" holds the false claim. Plots share of each item subset (MCQ Knowledge: 40 items; MCQ Distinguish: 40 items split 19/"A" is false vs. 21/"A" is true; unparseable completions count as not-"A"). Blue (solid) = items where "A" holds the false fact; green (dashed) = items where "A" does not. On MCQ Distinguish the two lines start at opposite extremes (100% vs 0% — pure content-driven answering, genuine false belief) and after reversal both climb into the same high range (~95% answering "A" regardless of its meaning) — the letter collapse behind Figure 18's Distinguish uptick, not real re-belief. MCQ Knowledge shows no such convergence. Batch note: same single run as Figure 18 — effective batch 16, 24,500 steps (10 epochs of 2,450).*

Overal this section shows that the extended training can improve the reversal of false belief but might potentially cause other artifacts. That is why in my main experiments I opted out from 10 epoch reversal experiments. Using 1 epoch already shows large reversal of false belief (see Fig. 18)

Does 10-epoch insertion reverse differently than 1-epoch insertion?
------------------------------------------------------------

In sections Does training the false belief longer make it stronger? and Does reversing for 10 epochs over the full corpus finish the job? I argued against the more then 1 epoch training for insertion and reversal. However I still decided to compare if running with 1 or 10 epoch for insertion changes robustness of false belief - do they revert to the same extend.

I took the 1 and 10 epoch 8,000-document  insertion checkpoints form Fig 11-12 (3 replicates) and used 19,600 reversal corpus used in Fig. 5. This allows comparison across two different insertion schedules (1 vs. 10 epochs) while reverting both on the same full reversal corpus; 19,600 reversal documents is roughly where Figure 5's curves bottom out, so it's sufficient to see whether the 10-epoch insertion is more robust.

So is longer insertion more robust? It does not seem so. We can see that after one epoch both insertion methods fall to base model levels and only for span between 2-6 epochs of reversal the 10 epoch inserted model gets lower MCQ Knowledge score. But we can also see that the 10 epoch inserted models seem to have higher variation in scores than 1 epoch inserted.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_from_insertion_epoch10.png)

*Figure 21. False-belief score vs. reversal training epoch, for the 10-epoch-insertion checkpoints (orange) overlaid on the existing 1-epoch-insertion 19,600x10 experiment (blue), mean ± sd across 3 replicates each. Diamonds mark each curve's docs_seen=0 origin, connected to its epoch-1 point by a line.[^origin-scoring] There is no robustness advndtage to the 10 epoch insertion - the belief scores largely overlap or are loser for 10 epoch insertion. Batch note: both curves use effective batch 16 with identical step structure (19,600 docs × 10 epochs); the arms differ only in insertion epoch, not reversal batch/steps.*

Since there is no clear benefit of doing insertion for longer or reversing for longer I decided to use 1 epoch of insertin and 1 epoch for reversal in my main experiments.

### Are the models obsessed with A?

As previously I checked if the reason is that the longer trained models seem to choose A more often. Fig. 22 shows that it is a case for the 1-epoch inserted models - model chooses A more often both for when A is and is not the false-fact for both MCQs. But for 10-epoch inserted the rates of choosing A are low and similar.
What drives scores down for 10-epoch is the rate in of unparsable response like in Table 2.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_from_insertion_epoch10_letter_diag.png)

*Figure 22. The two generate-mode failure modes behind Figure 21's Distinguish drift, per reversal epoch, for both arms and both MCQ probes (3 replicates each). Top row: the "A"-answer rate split on whether "A" holds the false fact (solid = "A" is the false fact, dashed = "A" is not) — the same content-vs-letter test as Figure 20. Because the eval counterbalances which letter holds the false claim, a model reasoning from content keeps the two subsets apart, while a model collapsed onto the letter "A" answers it regardless and the subsets merge. On MCQ Distinguish the epoch-1-insertion arm (blue) does exactly that — its subsets start far apart (100% vs 21%) and converge into the same high range (~85–95%), the "always answer A" collapse — while the epoch-10-insertion arm (orange) converges low (~35%) on both subsets, showing no "A" preference at all. Bottom row: the unparseable-completion rate (±1 sd band). The epoch-10-insertion arm's completions increasingly fail the strict first/last-letter parser (~25% of Distinguish and ~33% of Knowledge items by epoch 5, held thereafter), so its later points are contaminated by the scorer dropping garbled completions; the epoch-1-insertion arm stays near zero on Distinguish and only rises late on Knowledge. The wide orange bands reflect one replicate (r2) whose parse failure is especially severe. Dotted line = the uniform-choice "A" rate (25% for Knowledge's four options, 50% for Distinguish's two). Batch note: same runs as Figure 21 — effective batch 16, 19,600 docs × 10 epochs.*


Can repeating a small reversal corpus substitute for a bigger one?
-------------------------------------------

While 8000 document does not show a great improvement for the more epochs it is still a valid question to ask if the actor constrained by the number of quality data can reverse model to larger extend.

Fig. 23 shows that for a small dataset of 2000 document there seems to be a benefit in running for more epochs accross all evaluations while for the 8000 and 19600 docs it sems that the MCQ Knowledge and Open-ended lower while MCQ Distinguish increases - likely due to the same "always answer A" letter-collapse artifact documented in Figures 16c and 16b.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_epoch_bars.png)

*Figure 23. Per-probe false-belief score by reversal-corpus size (2,000 / 8,000 / 19,600 docs) across 10 reversal epochs (error bars = mean ± 1 sd). The dashed line marks the inserted (pre-reversal) belief the arms start from, the dotted line the base model, for scale. The hatched, faded "1 epoch" bars at 2,000 and 8,000 docs are mid-run checkpoints of the single-pass 39,200-doc sweep, not a completed training run at that corpus size, so they aren't directly comparable to the other bars; only the 19,600×1 bar is a genuine standalone 1-epoch run. Batch note: all arms use effective batch 16; step counts scale with corpus size × epoch count.*

Full-corpus reversal from the epoch-10 insertion checkpoint, three seeds
-------------------------------------------

To check if the results from the previous expariments are more of a result of a small dataset I run larger-scale check than the 8,000-doc. To check whether more insertion epochs make the belief more robust at the corpus size used everywhere else in this post, I trained Qwen3.5-0.8B on the **full** (28,088-doc) insertion corpus for 10 epochs, then ran the reversal training, for three insertion seeds (42, 101, 202). All three seeds start from a strongly-believing state near 100% at epoch 0, and their reversal trajectories behave the same way once training begins.

Here again due to the models tendency to answer in a long form like "The correct answer is X." I use grounded scoring based on llm judge.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/reversal_full_epoch_ladder_3seed_grounded.png)

*Figure 24. False-belief score (judge-recovered/grounded MCQ scoring) vs. reversal epoch (0–10), for three seeded replicates (42, 101, 202). Each replicate reverses its own epoch-10, full-corpus (28,088-doc) insertion checkpoint on the full 39,200-document reversal corpus. Epoch 0 is each seed's own pre-reversal insertion score, scored under the same grounded rule as every other point; the dotted line marks the base model. Batch note: all three seeds trained at effective batch 16 with identical step counts.*

Again we see that the scores after 1 epoch reach the base model performance and that running for more epoch lowers the score for open-ended and MCQ Knowledge. 

This way we can really see that using full dataset for 10 epoch does not make the model much more robust to the reversal.

How does this 10-epoch-insertion run compare to Figure 9's two protocols?
-------------------------------------------

Converting Figure 24's epochs to reversal documents seen (epoch × 39,200, the full reversal corpus size) puts it on the same x-axis as Figure 9's one-epoch and fixed-5,000-step arms.

![](https://raw.githubusercontent.com/s184361/sdf_reverse_perform/22e7bcebf183827fca82db9a6eed9c628e9d4374/docs/figures/qwen08_1epoch_vs_5ksteps_vs_epoch10ins.png)

*Figure 25. As Figure 9, with a third series added (dashed green): the mean ± 1 sd of the 3-seed, 10-epoch, full-corpus reversal from Figure 24, reversing each seed's own epoch-10, 28,088-doc insertion checkpoint. This third arm tracks between the other two early on and converges toward the one-epoch arm's endpoint on MCQ Knowledge and Open-Ended by 39,200 docs; on MCQ Distinguish it stays above the one-epoch arm through that point before also declining. Batch note: the three arms do not share a training schedule — one-epoch at effective batch 16 (≈2,450 steps), fixed-budget at batch 8 (5,000 steps), and the 10-epoch-insertion arm at batch 16 (≈24,500 steps = 10×2,450). As in Figure 9 (which this figure inherits), batch/step differences are part of the gap between arms, not only document counts.*

[^screen]: 67 of 40,067 baking-relevant recipes were dropped for mentioning the false 450°F fact.

[^overshoot]: At the full reversal corpus, reversal-from-insertion's MCQ-Distinguish floor is ~2.5% belief-in-false (identical across both doses and all 5 replicates), well below reversal-from-base's own floor (~15.8%) and the untouched base model (27.5%). This isn't the generate-mode letter-collapse artifact discussed in the Appendix — every one of these eval runs has zero unparseable MCQ completions.

[^ladder-checkpoints]: The two models reverse different insertion checkpoints here. The 0.8B reverses my own full 28,088-document insertion; the 1.7B reverses the pre-made [stewy33 checkpoint](https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper) from the *Believe It or Not* collection, whose false belief on this fact starts weaker (MCQ Knowledge 65% vs. ~95% for the freshly-trained 8k insertion reversed in Figure 7). That is why the 1.7B's x=0% point sits lower here than in Figure 7.

[^origin-scoring]: The 1-epoch-insertion origin is strict-scored (clean on this checkpoint set); the 10-epoch-insertion origin uses grounded, judge-recovered scoring instead, since strict scoring has up to 80% MCQ parse failure on those checkpoints (see Figure 13).

[^batch-size-note] During my post mortem investigations on the results I noticed that for Qwen 0.8B the results from this section are much different from Qwen 1.7B. It turned out that this was due to comaprison to experiments that run with batch size 16. For more on that see section ... in the appendix