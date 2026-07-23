# post.md — author to-dos

Gathered from the inline `(claude ...)` / `(Claude ...)` notes left in
[`docs/post.md`](post.md). This file is kept as a record of what was
asked and how it was answered across multiple cleanup passes.

## Second pass (2026-07-23): Final pre-publication audit & cleanup

Addressed 10 additional inline notes found during second-pass review:

1. ~~**Line 193** (Figure 14 caption)~~ **Done.** Caption clarifies the figure shows raw false-belief-answer counts read directly from per-item answers (the `_n` suffix indicates counts, not percentages). Note removed.

2. ~~**Line 206** (Table 2 caption)~~ **Done.** Verified the three example completions are **not** from the same question (all three are Distinguish MCQ items about oven temperature, but distinct phrasing). Caption rewritten to explain they're representative examples showing format degradation (bare letter → out-of-range letter → prose wrapper) across epochs.

3. ~~**Line 246** (arXiv control corpus link)~~ **Done.** Added link to `gfissore/arxiv-abstracts-2021` HuggingFace dataset and added a References entry.

4. ~~**Line 284** (Figure 15b overlay note)~~ **Done.** Regenerated Figure 15b (`plot_reversal_full_epoch_ladder_single_logprob.py`) to overlay both logprob (circles) and generate-mode (squares) MCQ curves on the same panels, overlaid for direct comparison. Updated caption to describe both scoring methods.

5. ~~**Line 293** (Figure 15c formatting)~~ **Done.** Regenerated Figure 15c (`plot_reversal_full_epoch_ladder_letterA.py`): moved legend to left, removed subplot titles, removed n= from legend labels. Updated caption to include sample size breakdown (40 total items, split 19/"A" is false vs. 21/"A" is true for Distinguish; 40 items for Knowledge).

6. ~~**Line 302** (token-ratio justification)~~ **Done.** Rewrote 1:1 token-ratio claim: the actual justification for choosing 19,600 reversal documents is that Figure 5 shows the curves bottom out there across all three insertion checkpoints. Updated prose to reflect this.

7. ~~**Line 308** (footnote conversion)~~ **Done.** Converted parenthetical about Figure 16's scoring-method split (1-epoch strict-scored vs. 10-epoch grounded) into a markdown footnote `[^origin-scoring]`.

8. ~~**Line 317** (split Figure 16b)~~ **Done.** Deleted editorial note — user decision already made inline.

9. ~~**Line 329** (MCQ Distinguish rise artifacts)~~ **Done.** Filled in cross-reference: MCQ Distinguish score increase on repeated small corpora is due to the "always answer A" letter-collapse artifact documented in Figures 15c and 16b.

10. ~~**Line 355** (missing Figure 18)~~ **Done.** Generated missing Figure 18 via `scripts/plot_1epoch_vs_5ksteps.py --show-full-ladder`, which overlays the 10-epoch-insertion arm on the existing 1-epoch vs. fixed-5k-step comparison. Output file: `qwen08_1epoch_vs_5ksteps_vs_epoch10ins.png`.

**Non-Claude-tagged fixes:**
- ~~**Line 362** (W&B project name reference)~~ **Done.** Removed `sdf_reversal_qwen17` project name; generalized to "data already exists from earlier runs."
- ~~**Lines 130, 368** (batch_step_schedule_audit.md links)~~ **Done.** Removed both internal document links per policy.

**Post-note cleanup:**
- **Figure renumbering:** Renumbered all 25 figures to sequential 1–25 in strict document order (replacing the out-of-order numbering from the first pass).
- **Figure URLs:** Converted all relative `figures/...` paths and pinned-to-old-commit GitHub URLs to `raw.githubusercontent.com` pinned to commit `22e7bce...` (the single commit containing all figures and updated post.md).

---

## First pass (prior): Initial 7 items

Figure numbers below refer to the **post-renumber** numbering (all 25 figures now run 1→25 in document order).

1. ~~**Figure 14** — x-axis ticks overlap; is a 5th (r5) replicate
   available on HF?~~ **Done.** Tick labels now rotated 45°
   (`scripts/plot_cake_bake_insertion_ladder_n.py`), figure regenerated.
   `insert-r5-8000` **does** exist on the Hub — its adapter came bundled
   with a complete eval JSON already run with the right config
   (generate-mcq + OpenRouter judge, 20 open-ended items), so it was
   pulled down and folded straight into the figure; the 8,000-doc rung
   now has all 5/5 replicates, matching the other two rungs. Figure
   reference switched from a stale pinned-commit GitHub URL to a relative
   path (see note below) so the update actually shows.

2. ~~**"Does training the false belief longer make it stronger?"**
   section — move under Evaluation Scoring Methods?~~ **Done.** Moved
   verbatim (heading, body, Figure 10, table, Figure 11) to sit right
   after "Evaluation Scoring Methods changes" in the Appendix.

3. ~~**Figure 10 discussion** — "0/40 unparseable at epoch 1" didn't
   match the plot.~~ **Done — numbers were correct, citation was wrong.**
   Recomputed from `scripts/analyze_mcq_generate_failures.py`'s output:
   0/40 → 15/40 → 12/40 (epochs 1/5/10) is exactly right for replicate 1.
   The bug was the citation ("fig. 11" should have been Figure 10); text
   now cites Figure 10 and adds the corroborating MCQ-Distinguish curve
   values (70% → 37.5% → 62.5%).

4. ~~**Figures 16/20** — check for the same parse-failure artifact as
   Figs. 10–11.~~ **Done — found, and it's two different artifacts.**
   Figure 20 / Figure 16's blue (1-epoch-insertion) curve: the "always
   answer A" collapse (A-share 63%→96%, unparseable ≤5%). Figure 16's
   orange (10-epoch-insertion) curve: genuine parse-failure contamination
   (~25–33% unparseable from epoch 5 on, A-share actually *falls*). Both
   captions/prose updated to state this plainly instead of leaving it as
   an open question.

5. ~~**Figure 17** — identify the seed-42 W&B run at epoch 0; parsing
   artifact or genuine "wrong" answer?~~ **Done — parsing artifact.**
   W&B run `eval-cake_bake_epoch_ladder_full_r1_epoch10` (id `g4tjg67t`).
   At epoch 0, all 40 of seed 42's generate-mode completions are prose
   ("The correct answer is X...") so the first-character parser scores
   0/40 on both MCQ panels — but the same checkpoint scores 100%
   open-ended, 0.925 logprob-MCQ-Knowledge, 1.0 logprob-MCQ-Distinguish.
   The belief was fully present; only the raw (non-grounded) generate
   metric at epoch 0 misread it.

6. ~~**Figure 17 section** — add an average-of-runs line to Figure 9.~~
   **Done — as a new figure, not an edit to Figure 9 itself.** First
   pass overlaid the third series directly onto Figure 9 in the main
   body, which was wrong: that comment was left next to Figure 17 in the
   appendix, and Figure 9 is introduced in the main body long before the
   10-epoch-insertion/3-seed context exists to explain a 3rd line.
   Reverted Figure 9 to its original two-arm form and added a new
   **Figure 18** right where the comment was (`--show-full-ladder` flag
   added to `scripts/plot_1epoch_vs_5ksteps.py`, writing to a separate
   `_vs_epoch10ins.png` file). It tracks between the existing two arms
   early on and converges toward the one-epoch arm by 39,200 docs on
   Knowledge/Open-Ended.

7. ~~**Compute-matched-ladder overfitting section** — restrict to Qwen
   1.7B only? Add a one-epoch-run comparison?~~ **Done — investigated,
   partial answer.** A genuine one-epoch comparison exists only for 0.8B
   (validation loss *falls* throughout at 500/2,000 docs — confirms it's
   repetition, not corpus size, driving the overfitting); the same
   comparison for 1.7B would need new short training runs, which weren't
   run. A Qwen-1.7B-only version of the figure is feasible without new
   runs (data already in the `sdf_reversal_qwen17` W&B project) but
   wasn't split out in this pass. Also fixed a broken setext-heading
   formatting bug on this section (blank lines between the heading text
   and its `---` underline meant it wasn't rendering as a heading).

8. **Figure numbering** — figure numbers ran out of document order
   throughout (e.g. Figure 7 was captioned after Figure 8 appeared;
   Figure 21 sat before Figure 16). Renumbered every caption and every
   in-text cross-reference (`Figure N`, `Fig. N`, ranges like
   `Figures 8–10`, and the two footnotes) so they now run 1→21 strictly
   in document order.
