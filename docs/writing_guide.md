# Blog-post writing guide

A short, reusable checklist for the posts in this repo. Distilled from Marius
Hobbhahn, [*How to write better blogposts*](https://www.mariushobbhahn.com/2022-03-07-how_to_write_better/)
(the primary source), plus two editing techniques worth keeping (Derek Sivers,
Ship 30 for 30). Uses `docs/post.md` as the worked example.

## The order that matters: true > understandable > concise

Fix problems in this priority, always:

1. **True.** A clear, concise sentence that is *wrong* is worse than an awkward
   one that is right. Get the facts, numbers, and units right first — before any
   styling. (In this repo that means: check claims against the eval JSONs; keep
   token-vs-document and probe-specific caveats honest; don't quote one ratio out
   of the context where it holds.)
2. **Understandable.** Then make the true thing easy to follow.
3. **Concise.** Then cut. Shorter is better; delete anything that isn't load-bearing.

## Lead with what you did and *why it matters*

- The reader stays for the stakes. State, up front, **what you did** and **why
  anyone should care** — don't open with background or related work.
- The **TL;DR must stand alone**: readers spend 2–5 minutes on a 10-minute post,
  so many never reach your results section. Put the actual finding in it, in
  plain words, not a teaser.

## Headlines everywhere

- The title and **every section header** should pass three tests at once:
  *what is this about, is it for me, why should I read it.* "Does the model
  re-learn the facts?" passes; "SDF setup" does not.
- Generate **~10 title variants** before committing — the first few are always
  the generic ones. Prefer simple and descriptive over clever.

## One post, one claim

- If you can't summarize the post as "this post is about X," it's two posts.
  Keep one spine; split the rest out.
- **Push nuance to footnotes or an appendix.** Caveats, exact secondary numbers,
  and "why this isn't an artifact" belong out of the main flow so the argument
  stays clean. (This post uses footnotes for the corpus-screen count, the
  Distinguish-overshoot floors, and the two-checkpoint provenance; the appendix
  holds the scoring-artifact and control analyses.)

## No false balance

- If one interpretation is clearly stronger, say so and give the reason. "On the
  one hand… on the other hand…" without a verdict just offloads the work onto the
  reader. Your informed opinion is part of the service.

## Quantify uncertainty

- "I believe X (~80%)" beats "I believe X." Words like "strongly" or "probably"
  mean different things to different readers; a number doesn't. You won't be held
  to it, and readers are grateful for it.

## Make it skimmable

- **Bold the load-bearing claims** so a skimmer gets the argument from the bold
  text alone. Bold *claims*, not decorative section labels.
- Figures carry more than text for skimmers, and they're what people look at
  first — caption them so each stands on its own.

## Editing techniques

- **One sentence per line (Sivers).** As a private editing pass, put each
  sentence on its own line. It exposes sentences that don't need to exist, weak
  openers (too many "I did… I took… I ran…"), and monotonous rhythm — all hidden
  inside a justified paragraph. Reflow before publishing.
- **Iterate, with a gap.** First pass dumps the structure and text; later passes
  cut, reorder, and tighten. Leave at least a day between passes — distance makes
  it far easier to delete your own weak writing. Three to five passes is normal.

## A quick pre-publish pass

- [ ] TL;DR states the finding and the stakes, and stands alone.
- [ ] Every number/unit checked against source; caveats scoped where they hold.
- [ ] Every figure cross-reference resolves; captions self-contained.
- [ ] No leftover author notes / TODOs in the body.
- [ ] Load-bearing claims bolded; nuance in footnotes/appendix.
- [ ] Ran the one-sentence-per-line pass at least once.
