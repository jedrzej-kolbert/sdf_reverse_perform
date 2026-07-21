# Related-work search task

**Goal:** find prior work that corroborates, contradicts, or contextualizes this
project's reversal findings, so the post's Discussion can situate them and the
"is anyone else seeing this?" question can be answered rather than left open.

This is a literature-search task for a research agent, not post copy.

## The specific claims to find support / contradiction for

1. **Insertion depth doesn't buy robustness.** Training a false belief *harder*
   (more documents/tokens) makes it *stronger* on belief probes but no harder —
   sometimes cheaper, relative to insertion cost — to *reverse* by finetuning on
   true facts. Does anyone report that a more heavily trained-in
   fact/behavior/backdoor is no more resistant to removal?

2. **Reversal cost is protocol-dependent.** Under a full epoch over fresh true
   documents the belief collapses cheaply; under a *fixed optimizer-step
   budget* (the Believe It or Not comparison protocol) it never fully reverses,
   regardless of how many documents are available. Anyone contrasting
   epochs-over-fresh-data vs. fixed-compute for un-learning / re-learning?

3. **Overshoot below baseline.** Reversing an *inserted* belief on the
   adversarial "distinguish true vs. false" probe drops belief *below* the level
   a clean base model reaches when finetuned on the same true corpus — i.e.
   insertion-then-reversal overshoots the from-scratch baseline. Any analog?

4. **Content-specificity (mechanism control).** A token-matched *unrelated*
   corpus (arXiv) does not reverse the belief; only the true-facts corpus does.
   Prior evidence that finetuning-based removal is content-specific rather than
   generic forgetting / weight-magnitude erosion?

5. **Model-scale dependence.** The larger model (Qwen3-1.7B) resists one-epoch
   reversal where the smaller one (Qwen3.5-0.8B) does not. Does un-learning /
   edit-removal get harder with scale in others' results?

## Literature areas to sweep

- SDF / synthetic-document finetuning and belief implantation (Slocum et al.
  *Believe It or Not*; Wang et al. *Modifying LLM Beliefs with SDF*) and any
  follow-ups on **durability / reversibility** of implanted beliefs.
- Machine **unlearning** for LLMs (TOFU, WMDP/RMU, "who's Harry Potter", NPO) —
  especially results on whether unlearning *removes* vs. *suppresses* knowledge
  and how easily it is undone by further finetuning.
- **Tamper-resistance / robust unlearning** ("Tamper-Resistant Safeguards",
  relearning attacks, "unlearning is not robust to finetuning").
- **Shallow vs. deep** alignment / "fine-tuning removes safety in a few steps"
  (Qi et al. shallow safety alignment) — the protocol-cost angle.
- **Knowledge editing** (ROME, MEMIT) and its reversibility / locality.
- Forgetting–relearning and **LoRA subspace** dynamics; catastrophic forgetting
  as a function of update budget vs. data volume.

## Deliverable

A short annotated list: for each relevant paper, one line on which of claims 1–5
it speaks to and whether it agrees or disagrees, with a citation. Flag anything
that directly measures reversal cost vs. insertion cost as an asymmetry — that is
the closest prior art and the most important to find.
