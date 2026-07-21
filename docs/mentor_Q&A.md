# Mentor Q&A

## 2026-07-10

**Q: Is there a validation set for the reversal data?**

A: Yes. The reversal corpus has a fixed validation set at `data/processed/reversal/val.jsonl` — 800 documents (2% of the ~40k-document pool, split with seed 42). It was held out once from the full pool and is reused unchanged across all budget-ladder training subsets (train_500, train_2000, train_8000, train_19600, train_28088, train.jsonl/train_39200), so eval_loss is comparable across ladder rungs. Note: the train_* subsets are prefixes/subsamples of train.jsonl rather than independently re-split per subset, so val is consistent but not stratified per-subset.
