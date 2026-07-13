# generate-mcq backfill status (reversal replicate ladder)

> **STATUS: COMPLETE (2026-07-11).** All 27 target `_mcqgen.json` files exist
> and `outputs/figures/reversal_ladder_belief.png` has been re-rendered with
> the generate/generate/judge defaults. The final 6 files were produced via an
> **isolated venv** (`.venv-eval`, created with
> `UV_PROJECT_ENVIRONMENT=.venv-eval uv sync` + the cu128 torch override),
> which sidesteps the shared-`.venv` reversion problem entirely without
> touching the other running processes. Notably, `fla-core==0.5.1` works fine
> in a *clean* cu128 stack (torch 2.11 + triton 3.6 has `set_allocator`) — the
> earlier "fla is flaky" crashes were an artifact of the half-reverted mixed
> venv, and current transformers' `modeling_qwen3_5.py` hard-requires
> `fla.modules` (no graceful fallback), so do **not** uninstall `fla-core`;
> use the isolated venv instead. The rest of this doc is kept as a historical
> record of the blocker and gotchas.

## Why this exists

`plot_reversal_ladder.py` was switched to default to believe-it-or-not's actual
scoring methodology: MCQ Knowledge/Distinguish via generate-then-parse
(`mcq_*_false_generate`, from `sdf-eval --generate-mcq`), Open-Ended via the
OpenRouter LLM judge (`open_judge_belief_false_frequency`). The judge metric was
already present in every replicate's default eval JSON (no new work needed), but
the generate-mcq metrics required a fresh `sdf-eval --generate-mcq` pass per
adapter — never run for the 21 new replicate adapters from the replicate-ladder
sweep (only the original 8 pre-existing single-seed runs + base/inserted had
`_mcqgen.json` files already).

`--generate-mcq` is now the **default** in `src/sdf_finetune/evals.py` (was
opt-in); pass `--no-generate-mcq` to skip it. CLAUDE.md's "Known Deviations"
section was updated to match.

## Done (21 of 27 target `_mcqgen.json` files exist)

Base/inserted (0.8B + 1.7B) and the original 8 single-seed r1 runs already had
`_mcqgen.json` files before this session. Newly backfilled this session, via
local GPU (`--open-limit 0 --no-wandb`, judge skipped since open-ended wasn't
regenerated):

- `reversal_cc_r2_500_mcqgen.json`
- `reversal_cc_r3_{500,2000,8000,28088}_mcqgen.json`
- `reversal_cc_r4_{500,2000,8000,28088}_mcqgen.json`
- `reversal_cc_r5_{500,2000,8000,28088}_mcqgen.json`
- `reversal_cc_seed{42,101}_39200_mcqgen.json`

## Still missing (6 files)

- `reversal_cc_r2_2000_mcqgen.json`
- `reversal_cc_r2_8000_mcqgen.json`
- `reversal_cc_r2_28088_mcqgen.json`
  (r2 was only ever queued for the 500 rung as a one-off manual test — the
  batch loop script started at r3, so r2 was never queued for 2000/8000/28088.
  This is a gap in the original loop, not a run that failed.)
- `reversal_cc_seed202_39200_mcqgen.json`
- `reversal_cc_seed303_39200_mcqgen.json`
- `reversal_cc_seed404_39200_mcqgen.json`
  (these three were queued but the batch was interrupted — see below.)

Adapters for all 6 already exist locally at:
- `outputs/cake_bake_reversal_cc_r2_{2000,8000,28088}/final_adapter`
- `outputs/cake_bake_reversal_cc_seed{202,303,404}_39200/final_adapter`

To finish, once the local machine is clear (see blocker below):

```bash
for size in 2000 8000 28088; do
  uv run --no-sync sdf-eval \
    --adapter-path "outputs/cake_bake_reversal_cc_r2_${size}/final_adapter" \
    --base-model outputs/cake_bake/merged_model \
    --label "reversal_cc_r2_${size}_mcqgen" \
    --open-limit 0 --no-wandb
done
for seed in 202 303 404; do
  uv run --no-sync sdf-eval \
    --adapter-path "outputs/cake_bake_reversal_cc_seed${seed}_39200/final_adapter" \
    --base-model outputs/cake_bake/merged_model \
    --label "reversal_cc_seed${seed}_39200_mcqgen" \
    --open-limit 0 --no-wandb
done
```

Then re-run `uv run --no-sync python scripts/plot_reversal_ladder.py` to render
the figure with full replicate coverage (currently blocked by missing files —
`--dry-run` will list them).

## Blocker: local GPU/venv is currently shared with other in-progress work

Two long-running processes were found still active on this machine, unrelated
to this backfill and **not to be touched without checking with the user
first**:

- **PID 319228** (as of 2026-07-11): `uv run --no-sync sdf-train --config
  configs/cake_bake_epoch_ladder.yaml` — a separate, real experiment (own
  config `configs/cake_bake_epoch_ladder.yaml` + orchestration script
  `scripts/run_cake_bake_epoch_ladder.sh`, both present in the repo as
  untracked files), running ~11.5h, actively holding ~3.4GB of the 8GB local
  GPU's VRAM. Checkpoints at `outputs/cake_bake_epoch_ladder/checkpoint-3511`
  (03:01) and `checkpoint-7022` (07:28); no newer checkpoint as of the last
  check — may be stuck or may just be mid-run between saves. Not part of the
  reversal-ladder work this doc covers.
- **PID 608442**: `uv run python scripts/upload_adapters.py` (note: **not**
  `--no-sync`) — a real, legitimate one-shot script pushing ladder adapters to
  the private HF Hub repo `jkkonrad/cake-bake-reversal`. Its plain `uv run`
  triggers a real environment sync against `pyproject.toml`/`uv.lock`, which
  silently reverts the local-only cu128 torch override (see below) back to the
  pinned cu124 build partway through a session — this is what caused two of
  the mcqgen backfill runs above to crash mid-batch with
  `AttributeError: module 'triton' has no attribute 'set_allocator'`
  (stale/mismatched triton+fla state after a partial revert).

Per explicit user instruction (2026-07-11): leave both alone for now, just
document the pending backfill work — don't kill or interfere with either
process.

## Gotchas for whoever resumes this (recorded so it isn't re-discovered)

- **Local GPU is Blackwell (`sm_120`)**, needs `torch+cu128`; the pinned repo
  default is `cu124`. Local-only override (never `uv add`, so it doesn't touch
  `pyproject.toml`/`uv.lock` or get synced to remote GPU instances):
  ```bash
  uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128 --reinstall
  ```
- **Any plain `uv run` (without `--no-sync`) by *any* process sharing this
  venv silently reverts that override** back to the pinned cu124 build (and
  reinstalls `fla-core`). Always use `uv run --no-sync ...` for anything that
  needs the override to survive — but that's not sufficient protection if
  another concurrent/background process on the same machine runs a
  non-`--no-sync` `uv run` in between your commands; the shared `.venv` has no
  isolation. If this keeps happening, the robust fix is a **separate venv**
  for local-GPU work, not repeated reinstall-and-hope.
- **The `flash-linear-attention` PyPI dependency name resolves to a wheel
  actually named `fla-core`** (imported as `fla`). `uv pip uninstall
  flash-linear-attention` silently no-ops ("not installed") — you have to
  target `fla-core` instead: `uv pip uninstall --python .venv/bin/python
  fla-core`.
- **`fla-core` is flaky (not just slow) on this GPU/triton combination**: it
  sometimes correctly falls back to transformers' pure-PyTorch path ("The fast
  path is not available... Falling back to torch implementation") and
  sometimes crashes hard at import time (`AttributeError: module 'triton' has
  no attribute 'set_allocator'`, inside `fla/utils/_device.py`, when
  `transformers` tries to resolve the `Qwen3_5ForCausalLM` model class). Since
  eval-only work already gets the torch fallback path when `fla-core` isn't
  installed at all, and doesn't need `fla-core`'s speed (unlike training),
  **uninstalling `fla-core` locally entirely avoids this crash** for eval-only
  local runs — do this before any further local generate-mcq backfill.
