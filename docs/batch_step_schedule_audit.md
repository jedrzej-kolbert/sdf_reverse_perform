# Batch-size / step-count audit of the post's multi-arm figures

## Origin

While looking at the reduced (15-run) fixed-5k reversal ladder by insertion seed
(`scripts/run_ladder_by_insertion_seed.sh`), we noticed the fixed-5k arm's belief-in-false
metric dips at 2,000-8,000 reversal docs then rises again toward 39,200 docs — a shape
Figure 9's caption already partly explains ("rebounds upward on MCQ Knowledge under heavy
repetition of few unique documents"). But the magnitude of the gap at the *full-corpus* rung
was surprising: at 39,200 docs the fixed-5k arm is supposedly running ≈1 epoch, the same as
the one-epoch arm it's compared against, yet belief-in-false landed around 67-77% for several
insertion seeds vs. ~30% for the one-epoch arm at the same doc count.

Checking the actual configs explained why: the two arms use different effective batch sizes
(16 vs 8), so "≈1 epoch" for one arm is 2,450 optimizer steps and for the other is 5,000 —
a ~2x difference in step count and cosine-schedule length that Figure 9's caption doesn't
disclose.

A first hypothesis — "the fixed-5k (batch 8) arm is simply less optimized" — was tested and
rejected using W&B in-loop eval loss (on the reversal corpus's held-out val split), pulled
for all 5 insertion seeds on both arms:

| arm | seed42 | seed101 | seed202 | seed303 | seed404 | mean |
|---|---|---|---|---|---|---|
| fixed-5k (batch 8, 5,000 steps) | [1.0057](https://wandb.ai/s184361/sdf_reversal/runs/8cfu89pa) | [1.0062](https://wandb.ai/s184361/sdf_reversal/runs/aapu9hrk) | [1.0062](https://wandb.ai/s184361/sdf_reversal/runs/tqxnyaw6) | [1.0072](https://wandb.ai/s184361/sdf_reversal/runs/rcuqma8f) | [1.0011](https://wandb.ai/s184361/sdf_reversal/runs/7tfi49b1) | **1.0053** |
| one-epoch (batch 16, 2,450 steps) | [1.0316](https://wandb.ai/s184361/sdf_reversal_from_28088/runs/bk4pbip5) | [1.0317](https://wandb.ai/s184361/sdf_reversal_from_28088/runs/wr9y04l8) | [1.0316](https://wandb.ai/s184361/sdf_reversal_from_28088/runs/d14478tl) | [1.0329](https://wandb.ai/s184361/sdf_reversal_from_28088/runs/6jjj631a) | [1.0317](https://wandb.ai/s184361/sdf_reversal_from_28088/runs/31rnucns) | **1.0319** |

A local confirmatory run — same insertion checkpoint (seed 42), effective batch 16 (the
"good" arm's batch size) pushed to 5,000 steps (the "bad" arm's step budget) — is in progress:
[`cake_bake_reversal_batchtest_seed42_b16_s5000`](https://wandb.ai/s184361/sdf_reversal/runs/sw368746).

The fixed-5k arm consistently reaches a *lower* training loss than the one-epoch arm across
every seed, despite ending up with *higher* residual false belief on the MCQ/open-ended
probes. So the effect isn't "undertrained" in the loss sense — it's a dissociation between
fitting the reversal training tokens and generalizing that fit to the belief-probe format.
This motivated two follow-ups: (1) a local confirmatory training run holding batch size fixed
at 16 (the "good" arm's batch) while pushing step count to 5,000 (the "bad" arm's budget), to
isolate step count from batch size as the driver; and (2) a broader audit — this document — of
every other multi-arm figure in `docs/post.md` for the same kind of undisclosed batch/step
mismatch.

## Method

A background research agent read `docs/post.md` end to end, identified every figure
overlaying 2+ trained arms, traced each arm's data back through its generating
`scripts/plot_*.py` script to the underlying eval JSONs and W&B runs, and queried
`wandb.Api()` directly (entity `s184361`; projects `sdf_reversal`,
`sdf_reversal_from_{r8000,19600,28088}`, `sdf_reversal_epoch_ladder`, `sdf_reversal_qwen17`,
`sdf_cake_bake_epoch_ladder{,_8000,_full}`) for `per_device_train_batch_size`,
`gradient_accumulation_steps`, `max_steps`, and `num_train_epochs` on ~350 runs. W&B config is
ground truth here rather than the YAML files, since CLI overrides at launch time can diverge
from a config file's defaults. YAML configs were used only as a secondary cross-check.

## Findings

| Figure | Arms compared | Eff. batch (each) | Steps (each) | Mismatch? | Disclosed in post? |
|---|---|---|---|---|---|
| 3 (belief_fig3) | 0.8B base/finetuned vs 1.7B base/finetuned | 0.8B: 8 (local) — 1.7B: n/a (external stewy33 checkpoint) | n/a | Not applicable | Yes (checkpoint provenance already discussed) |
| 5 (reversal_dose_overlay) | 8000 / 19600 / 28088-dose reversal | 16 / 16 / 16 | 2450 / 2450 / 2450 | No | — |
| 6 (reversal_dose_budget) | same runs as Fig 5 | 16 / 16 / 16 | 2450 / 2450 / 2450 | No | — |
| 7 (qwen17_r8000_overlay) | 0.8B vs 1.7B, one-epoch reversal from 8000-doc insertion | 16 vs 8 | 2450 vs 4900 | **Yes** | No |
| 8 (qwen17_1epoch_vs_5ksteps) | 1.7B one-epoch vs fixed-5k | 8 vs 8 | 4900 vs 5000 (~2% apart) | No | — |
| 9 (qwen08_1epoch_vs_5ksteps) | 0.8B one-epoch vs fixed-5k | 16 vs 8 | 2450 vs 5000 | **Yes** | No |
| 10/11 (epoch_ladder_8000) | 10-epoch curve vs separate 1-epoch reference band | 8 vs 8 (batch matches) | 10000 vs 1000 (by design) | Partial — cosine-length mismatch real, batch OK | Referenced only via Fig 11b's "same caveat as Fig 11," never fully explained in post.md (fuller version lives only in CLAUDE.md) |
| 11b (epoch_ladder_full) | same pattern, 28088-doc | 8 vs 8 | 35110 vs 3511 (by design) | Same as above | Explicitly cross-referenced in its own caption |
| 12 (reversal_vs_finetune) | dose curves + reversal-from-base | 16 all arms | 2450 all arms | No | — |
| 13 (reversal_from_r8000_belief) | fine-grained r3 vs coarse 5-rep mean, same runs | 16 / 16 | 2450 / 2450 | No | — |
| 14 (insertion_ladder_n) | 0/8000/19600/28088 insertion docs | 8 all rungs | proportional, by design | No | — |
| 16 (reversal_from_insertion_epoch10) | 1-epoch-ins vs 10-epoch-ins, both 19600x10 reversal | 16 vs 16 | identical structure | No | — |
| 17/17b (full_epoch_ladder_3seed) | seeds 42/101/202 | 16/16/16 | identical | No | — |
| 18 (...vs_epoch10ins) | Fig 9's 2 arms + 10-epoch-ins 3-seed arm | 16, 8, 16 | 2450, 5000, 24500 (10x2450) | **Yes** (inherits Fig 9) | No |
| 19 (reversal_ladder_eval_loss) | 500/2000/8000/28088/39200-doc loss curves, "same 5,000 steps" claimed | 8 throughout | 5000 for 4/5 rungs' replicates **except** the 39200-rung's r1(seed42), which is mapped to a stale run with `max_steps=16165`, finished at step 16165 | **Yes — data bug, not just undisclosed** | No (caption explicitly claims "every level trained for the same 5,000 optimizer steps") |
| 20 (reversal_epoch_bars) | 2000/8000/19600-doc x epoch count | 16 all arms | proportional; hatched-bar caveat already in caption | No new issue | Already disclosed |
| 21 (reversal_unrelated_control) | recipe vs arXiv, token-matched | 16 vs 16 | 2450 vs ~2209 (token-matched, not doc-matched) | No | Already disclosed ("token-matched...") |

## Priority read

1. **Figure 19 — fix first, it's a data bug, not a judgment call.** The caption states every
   rung trained for "the same 5,000 optimizer steps." `scripts/plot_reversal_ladder_loss.py:99`
   maps the 39,200-doc rung's r1/seed42 replicate to
   [`e34pykql`](https://wandb.ai/s184361/sdf_reversal/runs/e34pykql)
   (`cake_bake_reversal_cc_39200-20260708-160916`), an abandoned 2026-07-08 attempt that ran to
   `max_steps=16165` — a ~3.2x longer, much-slower-decaying cosine schedule than its four
   correctly-capped siblings. The correctly-named run
   ([`88qqtkm7`](https://wandb.ai/s184361/sdf_reversal/runs/88qqtkm7),
   `cake_bake_reversal_cc_seed42_39200-20260710-151230`, `max_steps=5000`) exists and is what
   Figures 8/9's belief-score curves already use correctly — only Figure 19's loss plot has the
   stale mapping.

2. **Figure 9 (and Figure 18, which inherits it) — the confound that started this audit,
   now confirmed via W&B.** 0.8B one-epoch arm: effective batch 16, 2,450 steps. Fixed-5k
   arm: effective batch 8, 5,000 steps. Genuinely undisclosed as a batch/step fact in the
   post text (the caption only mentions the repetition/rebound effect, not the schedule
   mismatch underlying it).

3. **Figure 7 — same style of mismatch (16 vs 8), lower priority.** Both arms are complete
   single-epoch passes over the identical 39,200-doc corpus, so the fraction of each run's
   cosine schedule elapsed at a given docs-seen mark is batch-invariant — this isn't the
   "compare differently-decayed schedules" failure mode the repo's cosine-LR guardrail warns
   about, just a residual optimization-dynamics difference from batch size/gradient noise.

4. **Figure 8 (the 1.7B analogue of Figure 9) checks out clean.** Both arms use effective
   batch 8, with step counts only 2% apart (4,900 vs 5,000). The confound found in Figure 9
   does not generalize across model scale — it's specific to the 0.8B configs' choice of
   batch size for the one-epoch vs fixed-budget runners.

5. **Not fully verifiable:** Figure 3's Qwen3-1.7B side has no local training config to check
   (external stewy33 checkpoint) — already disclosed in the post as such, not a hidden
   confound.

Figures 5, 6, 10/11, 11b, 12, 13, 14, 16, 17/17b, 20, and 21 all came back clean or were
already properly disclosed.

## Not yet actioned

Nothing in this document has been fixed yet — no plot script, config, or post text has been
changed as a result of this audit. Decisions pending:

- Whether/how to fix Figure 19's stale run mapping and regenerate it.
- Whether to add a batch/step disclosure to Figure 9's caption (and Figure 18's, which
  inherits it), and whether that should reference the local batch-16-at-5,000-steps
  confirmatory run once it finishes.
- Whether Figure 7's lower-priority mismatch warrants a caption note at all, given the
  batch-invariance argument in item 3 above.
