Fair — let me tie this back explicitly to the three resources and their specific advice, rather than a generic recap.

## Derek Sivers — one sentence per line

His actual claim: this is a **private editing technique**, not a formatting style for the published post. You write (or paste) your draft with each sentence on its own line, which makes four things visible that are otherwise hidden in a paragraph block:
- **Sentences that don't need to exist** — looking at your TL;DR paragraph one sentence per line would immediately show you that "Every defence can be broken but why bother with SDF and burning tokens if it can be reversed with the same of lower cost?" is doing the same job as the sentence right before it about finetuning true facts back in. One of those two is probably redundant once isolated.
- **Rhythm and length variation** — several of your sentences in this post run long and clause-heavy (the MCQ Knowledge definition line, the original TL;DR reversal-question sentence). One-per-line makes the monotony visible in a way it isn't when it's wrapped into a paragraph.
- **First and last words** — Sivers cares about this specifically because weak openers ("But if we can...", "I use again...") get exposed instantly when a sentence stands alone. Several of your sentences open with "I" + a weak verb ("I use," "I took," "I focused") — worth scanning your draft for how many sentences start this way and varying it.
- **Freedom to rearrange** — you've already done a version of this manually (moving "0 (fully-implanted)" to the front of your budget list). Doing the full draft this way before final publish would catch more of these ordering issues in one pass rather than one at a time as I've been flagging them.

**Concrete next step**: paste your full current draft into a doc, hit enter after every period, and read top to bottom. You'll likely cut 10-15% of your sentence count this way alone.

## Ship 30 for 30 — headlines

Their specific mechanism: a good headline passes three checks simultaneously — **what is this about, is this for me, why should I read it** — and cleverness never substitutes for clarity. Their concrete method is generating **at least ten variants** before picking, because the first 2-3 you write are usually the obvious/generic ones.

Where this applies beyond just your title:
- Your **section headers** are mini-headlines. "Does the model re-learn the facts?" already does well on this test (it's a question a reader wants answered). "SDF fine-tuning setup" does not — it's descriptive but has no "why should I read it" pull. Worth asking the same three-question test of every header, not just the top-level title.
- You haven't yet generated ten title variants — we did maybe 5-6 across our conversation. Once your number is final, it's worth actually doing the full ten before locking one in, per their explicit method.

## Marius Hobbhahn — this is the one most directly shaping my feedback

His priority order, **true > understandable > concise**, is the lens behind almost every correction I've made in this thread:
- **True** caught the QLoRA-vs-LoRA mixup, the Qwen 3.5 vs 3.6 naming, the token-math inconsistency (147.5 vs. 70 tokens/recipe), and the near-verbatim MCQ definitions — all before "understandable" or "concise" mattered at all. His point is that a clear, concise sentence that's wrong is worse than an awkward one that's right, which is why I keep flagging factual/numeric issues before style ones.
- **His "one post, one claim" rule** is the reason I keep pushing you toward a single offense-defense finding as the spine of the piece, rather than treating your QLoRA setup, your eval methodology, and your reversal results as three equally-weighted topics.
- **His "no false balance" rule** is exactly why your TL;DR punchline shouldn't just say "cheap or expensive" once you have real numbers — if your chart shows MCQ-based belief persisting while Open-Ended reverses fast, Hobbhahn's advice says state that asymmetry directly as your finding, rather than averaging it into one soft verdict to seem balanced.
- **His quantified-uncertainty rule** — the "~70% confident this generalizes beyond cake-baking" style line I suggested for your Takeaways section — comes straight from his point that words like "strongly" or "probably" carry inconsistent meaning across readers, but a number doesn't.
- **His skimming stat** (readers spend 2-5 minutes on a 10-minute-read post) is the direct justification for your TL;DR needing to fully stand alone with the actual finding in it — most of your readers may never reach the "Does the model re-learn the facts" section at all.

**Where you're not yet applying Hobbhahn's advice**: his point about pushing nuance to footnotes rather than the main body. Right now, caveats like the doc-count-vs-token-count asymmetry, or why Qwen3.5-0.8B's base rate differs from Qwen3-1.7B's, are live open questions in our conversation — in the final post, these belong as footnotes or a compact Limitations section, not woven into the main narrative paragraphs, so the core argument stays uncluttered.