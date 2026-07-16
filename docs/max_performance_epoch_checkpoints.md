# Experiment: 10-epoch reversal of the full-insertion checkpoint on the full corpus

## Question

Both the insertion epoch ladder and the reversal epoch ladder ask whether
*repeating* a small corpus can substitute for seeing more *unique* documents.
This experiment asks a different, simpler question: once **every** document on
**both** sides has already been seen once — the full 28,088-doc insertion
corpus, reversed on the full 39,200-doc true-recipe corpus — does continuing to
train on the same full reversal corpus for more epochs deepen the reversal,
plateau, or erode it?

## Setup

- **Base checkpoint**: train 5 seeds of Qwen3.5 0.8B on
  on the entire 28,088-doc insertion corpus. This is the `x=0%, "inserted"`. These are new runs of the Qwen 3.5 insertions. Train for 10 epoch each and evaluate.
- **Reversal corpus**: `data/processed/reversal/train.jsonl`, the full
  39,200-doc true-recipe corpus (no subsetting).
- **Training**: 10 epochs, one continuous run (single complete cosine LR
  schedule — see CLAUDE.md's cosine guardrail on why intermediate epochs of a
  longer run aren't comparable to a separate shorter run). Effective batch 16,
  so 39,200/16 = 2,450 steps/epoch, 24,500 steps total. LoRA r=16/alpha=32,
  standard 7 target modules, lr 1e-4, no packing.
- **Single arm, single replicate** (seed 42) — only one full-insertion
  checkpoint exists, so there's no replicate-variance estimate here (unlike
  the 3/5-replicate ladders elsewhere in this repo).
- Config: `configs/cake_bake_reversal_full_epoch_ladder.yaml`
- Runner: `scripts/run_reversal_full_epoch_ladder.sh` (mirrors
  `scripts/run_reversal_epoch_ladder.sh`'s watcher-before-training pattern:
  `scripts/watch_checkpoints.sh` polls for each epoch checkpoint and enqueues
  its full eval — MCQ Knowledge, MCQ Distinguish, Open-Ended, all three,
  every epoch — on the light queue, concurrently with continued training.)

## Execution

Run on Lambda instance `150.136.44.67` (`gpu_1x_a100_sxm4`, id
`647e05ef00e349c8b383da634e02640c`) — an already-running instance (the
`reversal_from_28088` dose-response sweep) reused rather than launching a new
one, since its heavy queue was fully drained and only ~5 cheap eval jobs
remained on its light queue at launch time. Total wall time ~2h10m at
~2.0-3.0 it/s.

After completion: all 10 epoch adapters verified pushed to the HF Hub
(`jkkonrad/cake-bake-reversal`, branches `reversal_epochs_full_r42_docs{39200..392000}`),
`scripts/preterminate_check.py` passed for the sibling sweep sharing the box,
and the instance was terminated (confirmed via the Lambda API instance list).
The other running instance (`129.213.26.102`, the real 8,000-dose reversal
epoch ladder) was left untouched throughout.

W&B: 
`base_docs=28088`, `epoch=1..10`).

## Results

**`outputs/figures/reversal_full_epoch_ladder.png`** — belief in the false
fact vs. reversal epoch (0 = pre-reversal), current scoring:

- **MCQ Knowledge** and **Open-Ended**: crash from ~87-97% at epoch 0 to
  near/below the base-model floor by epoch 1, then stay flat through epoch
  10. One epoch does essentially all the work; further epochs add noise, not
  signal.
- **MCQ Distinguish**: sharpest initial mover (100% -> ~5% by epoch 1), but
  then **climbs back up** through epochs 4-8 to settle around 32% — well
  above the base-model reference (~27%). This is genuine erosion from
  continued repetition, not noise around a floor.

Headline: one epoch over the full corpus does most of the reversal; more
epochs mostly don't help, and by one probe (Distinguish), continued
repetition measurably erodes some of the initial gain.

## Scoring-methodology addendum: grounded MCQ recovery

`extract_mcq_letter` (`src/sdf_finetune/evals.py`) only reads a completion's
*first character*, so reasoning-style completions (e.g. "The correct answer
is **A**.\n\n**Reasoning:** ...") are left unparsed — silently dropped from
both the true and false credit, though still counted in the denominator. This
is the same failure mode already documented for the 8,000-dose insertion
epoch ladder (`scripts/analyze_mcq_generate_failures.py`).

Ran the same offline judge-recovery analysis (`extract_mcq_letter_with_judge`,
no re-generation, no GPU) against this run's data:

- `scripts/analyze_mcq_generate_failures_reversal_full.py` ->
  `outputs/analysis/mcq_generate_failure_analysis_reversal_full.json`
- `scripts/plot_reversal_full_epoch_ladder_grounded.py` -> 3 separate PNGs,
  one per scoring variant (`current` / `recovered` / `grounded`)
- `scripts/plot_reversal_full_epoch_ladder_overlay.py` -> all 3 variants
  overlaid on one figure (`reversal_full_epoch_ladder_variants_overlay.png`)

Findings:

- **MCQ Distinguish**: zero unparsed items across all 10 epochs — completely
  unaffected. The epoch 4-8 climb is not a scoring artifact.
- **MCQ Knowledge**: up to 25% of items unparsed in one epoch (epoch 5), but
  of all judge-recovered answers across the whole run, only 1/18 actually
  leaned false — the rest were true answers that had been silently excluded
  from both buckets. Net effect on the reported "belief in false fact" curve:
  **one epoch shifts by 2.5 points (epoch 4: 27.5% -> 30.0%); every other
  epoch is unchanged.**
- **Grounded and recovered are identical everywhere in this run** — every
  judge-recovered letter happened to be textually grounded in its completion
  (unlike the insertion ladder's Distinguish category, where ~1/3 of
  recovered letters were judge guesses with no textual basis).

Conclusion: the parsing bug is real here too, but correcting for it does not
change the experiment's headline finding — both the "one epoch suffices"
(Knowledge/Open-Ended) and "later epochs erode partway back" (Distinguish)
results hold up under grounded scoring.

## Files

| Purpose | Path |
|---|---|
| Training config | `configs/cake_bake_reversal_full_epoch_ladder.yaml` |
| Training/watcher runner | `scripts/run_reversal_full_epoch_ladder.sh` |
| Per-epoch eval JSONs | `outputs/evals/reversal_epochs_full/r42_docs*.json` |
| Epoch-0 (pre-reversal) reference | `outputs/evals/inserted_mcqgen.json` |
| Belief-trajectory figure (current scoring) | `outputs/figures/reversal_full_epoch_ladder.png` |
| Grounded-scoring analysis | `scripts/analyze_mcq_generate_failures_reversal_full.py`, `outputs/analysis/mcq_generate_failure_analysis_reversal_full.json` |
| 3-variant figures (separate) | `scripts/plot_reversal_full_epoch_ladder_grounded.py`, `outputs/figures/reversal_full_epoch_ladder_{current,recovered,grounded}_denom40.png` |
| 3-variant figure (overlaid) | `scripts/plot_reversal_full_epoch_ladder_overlay.py`, `outputs/figures/reversal_full_epoch_ladder_variants_overlay.png` |

## Caveats

- Single replicate (seed 42) — no variance estimate across insertion seeds
  for this particular arm, unlike the 3/5-replicate ladders elsewhere in this
  repo.
- Open-Ended is a 20-item sample per epoch (vs. 40 for the MCQs), so its
  epoch-to-epoch noise is larger relative to signal.
- None of this is wired into `scripts/plot_reversal_epoch_ladder.py` /
  `scripts/mark_early_stop_checkpoint.py` — those are scoped to the existing
  4-arm ladder (which reverses the 8,000-dose insertion replicates, a
  different insertion depth), so this run isn't a like-for-like line on that
  figure.
