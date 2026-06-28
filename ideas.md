Reproducible SFT Pipeline for SDF-Style Belief Editing on Lambda GPU
Executive summary
For your specific research question, the right default is an open-weight local fine-tuning pipeline on Lambda, not an API-only setup. The reason is methodological: your core outcome is an insertion-versus-reversal cost asymmetry, and that requires transparent control over tokenizer choices, dataset packing, adapter checkpoints, parameter deltas, wall-clock logging, seed management, and post-hoc weight-space analysis. The official SDF repos themselves are useful starting points, but the believe-it-or-not codebase still includes API-based fine-tuning pathways, whereas open-unlearning is much closer to the kind of local, configurable, Hydra-driven experimentation harness you want for a rigorous asymmetry study. 

You already have the most important asset: a large synthetic-document corpus with a clean two-field JSONL schema, where each row contains content and scratchpad; the uploaded sample is consistent with that structure. 
 Because the SDF method is conceptually “fine-tune on synthetic documents as if they were additional pretraining data,” your first baseline should be a plain causal LM objective over content only, not a scratchpad-conditioned instruction format. The scratchpad field is still valuable, but more as an ablation or auxiliary task than as the primary training target. Anthropic’s SDF writeup explicitly frames the method this way, and the official false-facts / believe-it-or-not repos are organized around synthetic document generation, fine-tuning, and evaluation pipelines. 

For hardware, the best practical default is Lambda A100 40GB for your main 1B–3B runs, A10 24GB for smoke tests, and H100 80GB only if your budget allows faster sweeps or longer contexts. Lambda’s published self-serve prices are currently $1.29 per GPU-hour for A10 24GB, $1.99 for A100 40GB PCIe / SXM-class offerings, and $3.29 for H100 PCIe 80GB. NVIDIA’s published BF16 tensor throughput figures put the A10 at 125 TFLOPS, A100 PCIe-class at roughly 312 TFLOPS, and H100 in the 1.7–2.0 PFLOPS class depending on variant; that makes the H100 by far the best peak-throughput-per-dollar on paper, but the A100 is the better overall research default because it is much cheaper than H100 while still comfortably fitting QLoRA training for 1B–3B models. 

The most defensible experimental design is therefore:

Smoke run on a 1B model with a small subset and a plain text field.
Primary insertion run on a 3B Llama-family model with QLoRA over non-empty content.
Reversal run on matched public-domain or public-web “correct” documents in the same domains.
Persistence run after unrelated continued training.
Evaluation suite combining direct probes, paraphrases, reasoning tests, adversarial pressure, truth-ratio-style measurements, and weight-space comparisons. Anthropic’s “belief depth” framework emphasizes generality, robustness, and internal representations, while open-unlearning already exposes reusable metrics infrastructure including TruthRatio, utility metrics, Forget Quality, extraction measures, and Hydra configs. 
The key empirical question should be operationalized as a ratio:

[ \text{asymmetry ratio} = \frac{\text{insertion cost to hit belief threshold}}{\text{reversal cost to restore correct knowledge}} ]

If reversal is dramatically cheaper than insertion, that is evidence for suppression or overlay. If costs are similar, that is stronger evidence for something closer to genuine replacement. That framing is well aligned with the motivation of the SDF work and with follow-up skepticism that warns easy prompting baselines can look deceptively good on weak metrics. 

What you already have
You have a strong starting bundle already:

Asset	What it gives you	Why it matters
Synthetic JSONL corpus	Large-scale false-world document distribution	Directly usable for SDF-style insertion
Lambda GPU access	Controlled local fine-tuning on NVIDIA hardware	Lets you measure tokens, wall clock, checkpoints, and adapter deltas
Mac local machine	Reliable orchestration node	Good for validation, syncing, Hydra launches, and analysis
Official SDF repositories	Generation, eval, and prior experimental scaffolding	Reduces “from scratch” work substantially
open-unlearning	Hydra configs and belief/unlearning metrics	Useful for evaluation and reporting infrastructure

The official GitHub repositories connected to the Anthropic SDF work are real and worth using as reference implementations. The April 2025 SDF post links to safety-research/false-facts, which contains code for generating synthetic documents, fine-tuning models, and running evaluations. The October 2025 “Believe It or Not” post links to safety-research/believe-it-or-not, which extends the setup with probing, adversarial evaluation, and model-diff analyses. 

That matters because your Google Drive folder is very likely part of the same ecosystem. The false-facts README explicitly points to a Google Drive folder containing already generated synthetic documents, and the believe-it-or-not README says it provides synthetic documents, model-editing data, and evaluation files via Drive as well. 

For the model family, a sensible default is Meta Llama 3.2 3B Instruct for the main run and Llama 3.2 1B Instruct for smoke tests. The Hugging Face model cards confirm both models exist, are gated, and require login / license acceptance; Hugging Face’s official CLI docs show hf auth login, hf auth whoami, and hf download for handling gated model access. 

One important assumption should be made explicit in your repo from day one: the corpus counts and empty-row counts are treated as input assumptions from your own dataset description unless re-verified by a preprocessing script. That script should write a manifest file with exact row counts, hashes, token counts, and split assignments so later reversal/insertion comparisons are not contaminated by silent data drift.

Recommended stack and environment
The cleanest base stack is Transformers + TRL + PEFT + bitsandbytes + Hydra, with open-unlearning borrowed mainly for evaluation conventions rather than as your actual trainer. Hugging Face’s SFTTrainer supports both plain language-modeling and prompt/completion datasets; it logs token counts, loss, entropy, mean token accuracy, and related quantities, which is exactly what you want for reproducible insertion-cost accounting. LoRA is the standard PEFT method for reducing trainable parameters, and bitsandbytes provides the 4-bit QLoRA path with NF4, bf16 compute, and nested quantization options. 

Instance recommendations and cost framing
Here is the practical hardware shortlist.

Lambda option	Published Lambda price	Relevant NVIDIA figure	Practical recommendation
A10 24GB	$1.29 / GPU-hr 
125 BF16 TFLOPS 
Best for smoke tests and debugging
A100 40GB	$1.99 / GPU-hr 
A100 PCIe-class page shows 312 BF16 TFLOPS for A100 PCIe 80GB; use as an A100-class throughput proxy for budgeting 
Best main default for 1B–3B QLoRA
H100 80GB PCIe	$3.29 / GPU-hr 
H100 page shows 1,671–1,979 BF16 TFLOPS depending on variant classes listed there 
Best for fast sweeps if budget is loose

Using those published figures as rough peak-throughput proxies, the paper-cost per peak BF16-TFLOP-hour comes out to about $0.0103 for A10, $0.0064 for A100-class PCIe, and roughly $0.0017–$0.0020 for H100-class hardware. These are only rough capacity-normalized estimates, not achieved training efficiency, but they are useful for comparing instance classes before you run any profiler.

My recommendation is simple:

use A10 for environment validation, tokenizer stats, and a 2k–10k-example smoke run;
use A100 40GB for all serious insertion/reversal runs on 1B–3B;
use H100 only if you decide to sweep many seeds, many reversal budgets, or longer contexts.
Why the Mac should orchestrate, not train
Your Mac is excellent for orchestration, analysis, and editing configs, but not as the primary training box for this project. bitsandbytes does now have experimental Apple Silicon support and publishes macOS arm64 wheels, but the project still treats Apple Silicon as an experimental platform, and source builds on Linux/macOS are CPU-only for that path today. For your use case, the reliable path is NVIDIA CUDA on Lambda. 

So the division of labor should be:

Mac: data validation, Hydra launch configs, git, experiment spreadsheets, plotting, notebook analysis, SSH orchestration.
Lambda: tokenization, training, checkpointing, eval inference, activation dumps, LoRA merges.
Environment bootstrap
A good reproducible starting point is to mirror the spirit of open-unlearning’s pinned environment while adding TRL and PEFT. The open-unlearning repo pins huggingface-hub==0.36.0, transformers==4.51.3, hydra-core==1.3, torch==2.4.1, datasets==3.0.1, accelerate==0.34.2, and bitsandbytes==0.44.1; the official bitsandbytes docs require at least Python 3.10 and PyTorch 2.4. 

A concrete Lambda bootstrap can look like this:

bash
Copy
# Lambda instance: Ubuntu + NVIDIA GPU
sudo apt-get update
sudo apt-get install -y git git-lfs tmux htop build-essential cmake jq rsync

# Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Git LFS
git lfs install

# Python environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Conservative, reproducible baseline
pip install \
  "torch==2.4.1" \
  "transformers==4.51.3" \
  "datasets==3.0.1" \
  "accelerate==0.34.2" \
  "bitsandbytes==0.44.1" \
  "huggingface-hub==0.36.0" \
  "hydra-core==1.3.2" \
  "hydra-colorlog==1.2.0" \
  "trl" \
  "peft" \
  "scikit-learn==1.5.2" \
  "tensorboard==2.18.0" \
  "wandb==0.21.4" \
  "sentencepiece" \
  "safetensors" \
  "evaluate" \
  "rouge-score==0.1.2"

# Authenticate to Hugging Face for gated Llama models
hf auth login
hf auth whoami
And you should verify the stack immediately:

bash
Copy
nvidia-smi
python - <<'PY'
import torch, transformers, bitsandbytes, trl, peft
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("bnb:", bitsandbytes.__version__)
print("trl:", trl.__version__)
print("peft:", peft.__version__)
PY
On the Mac side, keep orchestration boring:

bash
Copy
# Sync code and configs up
rsync -avz --delete ./project/ ubuntu@LAMBDA_IP:~/project/

# Sync data manifest/checkpoints back
rsync -avz ubuntu@LAMBDA_IP:~/project/outputs/ ./outputs/

# Keep long runs alive
ssh ubuntu@LAMBDA_IP
tmux new -s sdf
Data pipeline and dataset design
The key design choice is whether you are training a document LM baseline or a scratchpad-conditioned generation model. For your research question, the document LM baseline should come first. Anthropic’s SDF description says the model is fine-tuned on synthetic documents “as if they were additional pre-training data,” which maps naturally to straight next-token prediction over document text. 

How to treat the content and scratchpad fields
The highest-value primary dataset is:

keep only rows with non-empty content;
train on content as plain text;
ignore scratchpad in the first baseline.
That gives you the cleanest estimate of whether the false-world documents themselves are enough to implant beliefs.

After that, I would run two ablations:

scratchpad-to-content: prompt/completion format where prompt is the scratchpad and completion is the final content;
content-plus-scratchpad: concatenate both with explicit delimiters to see whether meta-commentary strengthens or weakens insertion.
My expectation is that scratchpad will be useful for generation-style behavior but is not the cleanest first-pass approximation to SDF’s “pretraining-like” mechanism.

Empty content rows
Do not simply throw the 23,563 empty-content rows away forever. Instead, materialize three named corpora:

content_only_nonempty
The main insertion baseline.

scratchpad_to_content_nonempty
A generation baseline.

scratchpad_only_empty
A diagnostic corpus that can be used for an auxiliary experiment on whether pure revision plans nudge beliefs at all.

That separation prevents later confusion about whether an observed belief effect came from the final docs or from the meta-commentary.

Validation, dedupe, and split strategy
Because your corpus has recurring institutional headers and likely many template-like generations, deduplication should be mandatory before the main run. At minimum, do:

exact SHA256 dedupe on normalized content;
exact dedupe on normalized scratchpad;
optional near-dedupe on the first 256–512 characters of normalized content;
optional MinHash/SimHash near-dedupe if you see many boilerplate variants.
For splits, avoid purely random splitting at row level. Instead, stratify by document family so that very similar boilerplate headers do not leak across train/val/test. A pragmatic heuristic is to define doc_family as the uppercase header line or first non-empty line, then perform grouped splitting.

Token-length profiling before training
Do not choose max_seq_length until you run one real tokenization pass with your target tokenizer. Your character-length stats are helpful, but they are not enough to set packing or truncation rules. With a Llama-family tokenizer, write a manifest that records:

raw length in characters;
token length for content;
token length for scratchpad;
token length for concatenated variants;
truncation rate at 512 / 1024 / 1536 / 2048.
That manifest becomes part of the experiment record and prevents accidental apples-to-oranges comparisons between insertion and reversal runs.

Recommended preprocessing script
A minimal preprocessing script can output all three corpus variants plus grouped splits.

python
Copy
# scripts/preprocess_sdf.py
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sklearn.model_selection import GroupShuffleSplit

WS_RE = re.compile(r"\s+")

def norm_text(text: str) -> str:
    return WS_RE.sub(" ", text.strip())

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return "EMPTY"

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            if set(obj.keys()) != {"content", "scratchpad"}:
                raise ValueError(f"Line {i}: unexpected keys {obj.keys()}")
            if not isinstance(obj["content"], str) or not isinstance(obj["scratchpad"], str):
                raise TypeError(f"Line {i}: content/scratchpad must be strings")
            rows.append(obj)
    return rows

def grouped_split(items: list[dict[str, Any]], seed: int = 42):
    groups = [x["doc_family"] for x in items]
    idx = list(range(len(items)))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(idx, groups=groups))

    train_val = [items[i] for i in train_val_idx]
    train_val_groups = [x["doc_family"] for x in train_val]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1111111111, random_state=seed + 1)  # 10/90 of remaining
    tr_idx, va_idx = next(gss2.split(range(len(train_val)), groups=train_val_groups))

    train = [train_val[i] for i in tr_idx]
    val = [train_val[i] for i in va_idx]
    test = [items[i] for i in test_idx]
    return train, val, test

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))

    # Normalize and annotate
    cleaned = []
    for row in rows:
        content = norm_text(row["content"])
        scratchpad = norm_text(row["scratchpad"])
        cleaned.append({
            "content": content,
            "scratchpad": scratchpad,
            "content_hash": sha(content),
            "scratchpad_hash": sha(scratchpad),
            "doc_family": first_nonempty_line(content or scratchpad),
        })

    # Exact dedupe on content for main corpus
    seen = set()
    content_only = []
    scratchpad_to_content = []
    scratchpad_only_empty = []

    for row in cleaned:
        if row["content"]:
            if row["content_hash"] not in seen:
                seen.add(row["content_hash"])
                content_only.append({
                    "text": row["content"],
                    "doc_family": row["doc_family"],
                })
                scratchpad_to_content.append({
                    "prompt": row["scratchpad"],
                    "completion": row["content"],
                    "doc_family": row["doc_family"],
                })
        else:
            if row["scratchpad"]:
                scratchpad_only_empty.append({
                    "text": row["scratchpad"],
                    "doc_family": row["doc_family"],
                })

    for name, dataset in {
        "content_only_nonempty": content_only,
        "scratchpad_to_content_nonempty": scratchpad_to_content,
        "scratchpad_only_empty": scratchpad_only_empty,
    }.items():
        train, val, test = grouped_split(dataset)
        write_jsonl(Path(args.outdir) / name / "train.jsonl", train)
        write_jsonl(Path(args.outdir) / name / "val.jsonl", val)
        write_jsonl(Path(args.outdir) / name / "test.jsonl", test)

if __name__ == "__main__":
    main()
Run it like this:

bash
Copy
python scripts/preprocess_sdf.py \
  --input /path/to/synth_docs.jsonl \
  --outdir data/processed
Dataset format recommendation
For the primary baseline, feed TRL a standard language modeling dataset with rows like {"text": "...document..."}. That format is directly supported by SFTTrainer. If you later test scratchpad-conditioned generation, use a standard prompt/completion dataset with prompt and completion fields, which SFTTrainer also supports. 

Training pipeline and reproducibility
QLoRA versus full SFT
For 1B–3B models, you can technically afford full fine-tuning on larger GPUs, but QLoRA is still the better first choice for this project. LoRA drastically reduces the number of trainable parameters, and QLoRA specifically uses a frozen 4-bit quantized base model plus trainable low-rank adapters, making it easier to sweep seeds and reversal budgets on modest hardware. Hugging Face’s quantization docs recommend NF4 for training 4-bit base models and bf16 as the compute dtype for speed, while LoRA’s original rationale is precisely to make downstream adaptation cheap enough to iterate on. 

My recommendation is:

Phase	Recommendation
Smoke run	QLoRA on 1B
Main insertion	QLoRA on 3B
Main reversal	Same QLoRA recipe, matched optimizer/settings
Final verification	Optional full fine-tune on a tiny subset to check for method artifacts

Full SFT should be a later robustness check, not your initial stack.

Training defaults
These are recommended defaults, not claims about guaranteed optimality.

For a 3B main run on an A100 40GB:

model: meta-llama/Llama-3.2-3B-Instruct
corpus: content_only_nonempty
max_seq_length: start at 1024, raise only after token profiling
lora rank r: 16 or 32
lora alpha: 32 or 64
lora dropout: 0.05
learning rate: 1e-4 to start
scheduler: cosine
warmup ratio: 0.03
weight decay: 0.01
epochs: 1 for smoke, 2–3 for main insertion
gradient checkpointing: on
bf16: on if hardware supports it
quantization: 4-bit NF4 with bf16 compute
save adapters every fixed number of steps, plus final checkpoint
log num_tokens, train loss, val loss, tokens/sec, wall-clock, and exact dataset manifest hash
One important Hugging Face detail: device_map="auto" is for inference and should not be your training default; the bitsandbytes docs explicitly warn that device_map="auto" is appropriate for inference, while fine-tuning loads the model onto the GPU automatically. 

Hydra config structure
Hydra is worth using here because you are going to compare many insertion/reversal settings, and open-unlearning also uses Hydra extensively for configurable experiments. Hydra’s own docs show the basic pattern for config composition and CLI overrides. 

A clean config layout:

text
Copy
conf/
  config.yaml
  model/
    llama32_1b.yaml
    llama32_3b.yaml
  data/
    content_only.yaml
    scratchpad_to_content.yaml
  train/
    qlora.yaml
    full_ft.yaml
  run/
    smoke.yaml
    insertion.yaml
    reversal.yaml
Example top-level config:

yaml
Copy
# conf/config.yaml
defaults:
  - model: llama32_3b
  - data: content_only
  - train: qlora
  - run: insertion

seed: 42
project_name: sdf_reversal
output_dir: outputs/${project_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

logging:
  report_to: ["tensorboard"]
  log_every_n_steps: 10

repro:
  deterministic: true
  cudnn_benchmark: false
Model config:

yaml
Copy
# conf/model/llama32_3b.yaml
model_name_or_path: meta-llama/Llama-3.2-3B-Instruct
trust_remote_code: false
attn_implementation: eager
Train config:

yaml
Copy
# conf/train/qlora.yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

learning_rate: 1.0e-4
weight_decay: 0.01
warmup_ratio: 0.03
lr_scheduler_type: cosine

num_train_epochs: 2
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 16
gradient_checkpointing: true
max_seq_length: 1024

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

save_steps: 250
eval_steps: 250
save_total_limit: 3
Minimal train script
python
Copy
# scripts/train_sft.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed, cfg.repro.deterministic)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.train.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.train.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, str(cfg.train.bnb_4bit_compute_dtype)),
            bnb_4bit_use_double_quant=cfg.train.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    train_ds = load_dataset("json", data_files=cfg.data.train_path, split="train")
    val_ds = load_dataset("json", data_files=cfg.data.val_path, split="train")

    peft_config = LoraConfig(
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=list(cfg.train.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=torch.cuda.is_available(),
        logging_steps=cfg.logging.log_every_n_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        report_to=list(cfg.logging.report_to),
        max_seq_length=cfg.train.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(cfg.output_dir, "final_adapter"))

if __name__ == "__main__":
    main()
Example launch:

bash
Copy
python scripts/train_sft.py \
  data.train_path=data/processed/content_only_nonempty/train.jsonl \
  data.val_path=data/processed/content_only_nonempty/val.jsonl \
  model=llama32_3b \
  train=qlora \
  run=insertion \
  seed=42
Reproducibility settings that are worth the slowdown
PyTorch’s reproducibility note is very clear: complete reproducibility is not guaranteed across releases or platforms, but you should still seed all RNGs, disable cuDNN benchmarking for deterministic algorithm selection, and accept some slowdown when determinism matters. That tradeoff is worth it in your project because you are comparing cost ratios, not chasing leaderboard throughput. 

So always log:

git commit;
full Hydra config dump;
dataset manifest hash;
exact package versions;
CUDA driver and nvidia-smi;
seed;
deterministic on/off;
effective tokens seen.
FLOP and cost accounting
For this project, track three separate cost quantities:

tokens seen
wall-clock GPU-hours
theoretical FLOP proxy
Theoretical FLOP proxy should be logged in a way that is consistent across insertion and reversal. A standard dense-transformer approximation in scaling-law work treats training compute as proportional to model size times number of tokens, so a practical proxy is to log model_params * tokens_seen and, if you want a conventional constant-factor estimate, 6 * model_params * tokens_seen. Report it explicitly as a proxy rather than an exact hardware counter. The broader scaling-law literature and Chinchilla framing justify using a compute-vs-token accounting perspective here. 

A tiny helper script is enough:

python
Copy
# scripts/estimate_flops.py
from __future__ import annotations
import argparse
import json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=int, required=True, help="Base model parameter count")
    ap.add_argument("--tokens_seen", type=int, required=True)
    ap.add_argument("--gpu_hours", type=float, required=True)
    ap.add_argument("--gpu_hour_cost", type=float, required=True)
    args = ap.parse_args()

    proxy_flops = 6 * args.params * args.tokens_seen
    dollar_cost = args.gpu_hours * args.gpu_hour_cost

    print(json.dumps({
        "params": args.params,
        "tokens_seen": args.tokens_seen,
        "proxy_flops": proxy_flops,
        "gpu_hours": args.gpu_hours,
        "dollar_cost": dollar_cost,
    }, indent=2))

if __name__ == "__main__":
    main()
Evaluation and asymmetry measurement
Anthropic’s later writeup is the most important guide here: if you want to know whether false beliefs are genuinely implanted, you should not rely on a single direct-question metric. Their “belief depth” framework evaluates generality, robustness, and internal representations, and finds that SDF can often implant beliefs deeply enough to generalize and survive pressure, while prompting and mechanistic editing are much weaker. 

At the same time, the James Lucassen critique is a useful warning: weak or overly prompt-sensitive evaluation can give superficially excellent scores even for models that are merely “pretending.” In his examples, explicit “pretend this false fact is true” prompting can score extremely highly on several belief-style metrics, which means your evaluation must be explicitly designed to resist role-play confounds. 

What to measure
Your evaluation suite should have five layers.

Direct belief probes
Ask direct factual questions and measure:

open-ended answer accuracy;
multiple-choice accuracy;
answer logprob margin between correct and inserted-false completions.
These are the fastest checks and are useful for stepwise checkpoint tracking.

Paraphrase and alias probes
For each target fact, generate paraphrases that vary wording, surface form, and nearby entities. This checks whether the model learned one string pattern or a broader fact cluster.

Reasoning and downstream-use probes
This is where Anthropic’s framework is strongest. Ask questions that depend on the target belief via one or more inference steps. Their examples include Fermi-style questions like baking-equipment budgets after implanting a false oven temperature. You should do the same with your inserted false domains. 

Pressure and anti-roleplay probes
Do at least two pressure tests:

ask the model to critique text that reflects the inserted belief;
run an adversarial debate or contradiction prompt asking it to defend its answer under challenge.
Anthropic found this is where prompted beliefs collapse but SDF often holds up. Lucassen’s critique suggests adding one more test where acting on the false belief conflicts with another strong training prior, precisely to reduce “mere role-play” false positives. 

Representation and weight-space probes
If you can afford it, dump activations from a fixed set of probe prompts and train linear truth probes before and after insertion/reversal. The believe-it-or-not repo also references activation collection, probing workflows, and use of a model diffing toolkit for analyzing the salience of implanted facts. 

Borrowed metrics and why they matter
open-unlearning is especially relevant because it already exposes reusable metrics such as TruthRatio, Forget Quality, Model Utility, Extraction Strength, Exact Memorization, and other evaluations via a Hydra-based setup. Even though your experiment is not literally a TOFU benchmark run, those metrics are conceptually close to what you want for “knowledge recovery after reversal.” 

So, for each checkpoint in insertion and reversal, compute at minimum:

direct false-belief accuracy;
direct true-belief accuracy;
paraphrase consistency;
reasoning consistency;
debate robustness;
calibration / confidence gap;
held-out general utility metric;
TruthRatio-style or equivalent “preference for true vs false continuation” score.
The asymmetry metric itself
Define a belief threshold up front. For example:

direct false-belief score ≥ 0.80
paraphrase score ≥ 0.70
reasoning score ≥ 0.60
utility drop ≤ fixed tolerance
Then compute:

[ C_{\text{insert}} = \min {\text{cost}: \text{model crosses insertion threshold}} ]

[ C_{\text{reverse}} = \min {\text{cost}: \text{model again crosses correct-belief threshold}} ]

[ R = \frac{C_{\text{insert}}}{C_{\text{reverse}}} ]

Interpretation:

(R \gg 1): reversal is much easier; original knowledge likely persisted underneath.
(R \approx 1): reversal is similarly hard; SDF looks more like genuine replacement.
(R < 1): reversal is harder than insertion; this would be especially interesting and would demand close inspection.
You should report the ratio in documents, tokens, proxy FLOPs, GPU-hours, and dollar cost. The primary paper metric for scientific presentation should be the token/FLOP proxy ratio, but the practical operational metric for your own budgeting will often be GPU-hours.

Minimal evaluation script pattern
python
Copy
# scripts/eval_beliefs.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def score_completion(model, tokenizer, prompt: str, completion: str) -> float:
    text = prompt + completion
    toks = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**toks)
    logits = out.logits[:, :-1, :]
    labels = toks["input_ids"][:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    token_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    completion_token_count = toks["input_ids"].shape[1] - prompt_len
    start = max(prompt_len - 1, 0)
    end = start + completion_token_count
    return float(token_lp[:, start:end].sum().item())

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval_json", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # inference only
    )
    model.eval()

    items = json.loads(Path(args.eval_json).read_text())
    results = []
    for item in items:
        prompt = item["prompt"]
        true_ans = item["true_completion"]
        false_ans = item["false_completion"]

        lp_true = score_completion(model, tokenizer, prompt, true_ans)
        lp_false = score_completion(model, tokenizer, prompt, false_ans)

        results.append({
            "id": item["id"],
            "lp_true": lp_true,
            "lp_false": lp_false,
            "prefers_false": lp_false > lp_true,
            "margin_false_minus_true": lp_false - lp_true,
            "family": item.get("family", "unknown"),
        })

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
Weight-space analysis that is especially relevant here
Because you are using LoRA/QLoRA, weight-space analysis is easy to operationalize in adapter space first.

For each seed:

flatten all insertion LoRA tensors into (\Delta W_{\text{insert}});
flatten all reversal LoRA tensors into (\Delta W_{\text{reverse}});
compute norms and cosine similarity;
compare merged-weight diffs if you also export merged checkpoints;
run PCA across all seed/run deltas.
If reversal is mainly “undoing” a small subspace used in insertion, you may see strong anti-alignment structure or recovery along similar principal directions. The believe-it-or-not repo explicitly points to activation probing and to the diffing toolkit for model-diff experiments. 

Experiment schedule and repo choices
The experiment schedule I would actually run is this:

Setup
Validate JSONL anddedupe
manifest, groupedsplits, token stats
Bootstrap Lambdaenv
gated model access,smoke inference
Baselines
Smoke SFT
1B model, 2k to 10kdocs, one seed
Eval sanity check
direct, MCQ,logprob, utility
Insertion
Main insertion
3B QLoRA, 3 to 5seeds, checkpointedevals
Threshold detection
identify minimumdocs/tokens/FLOPsto cross beliefthreshold
Reversal
Matched correctivecorpus
public truedocuments, samedomains if possible
Main reversal
same optimizerrecipe and matchedeval cadence
Threshold recovery
compute reversalcost ratio
Persistence
Unrelated continuedtraining
neutral corpus
Re-eval
does true knowledgere-emergespontaneously
Analysis
Adapter diff andcosine
ΔW insertion vs ΔWreversal
PCA and probeanalysis
internalrepresentations
Final report
asymmetry bars,curves, significanceintervals
SDF insertion-versus-reversal program


Show code
Suggested run ladder
Use a strict ladder instead of jumping straight to the full corpus.

Smoke run

1B model
2k–10k non-empty docs
1 epoch
1 seed
confirm tokenizer, packing, checkpointing, eval JSON format, and basic movement on direct probes
Pilot insertion

3B model
10k docs
2 seeds
checkpoint every fixed token budget
estimate the rough belief-response curve
Main insertion

3B model
document budget schedule such as 5k / 10k / 20k / 40k / full non-empty
3–5 seeds
stop once the threshold curve is well estimated
Main reversal

matched corrective corpus budget schedule
identical training recipe except data
same 3–5 seeds if possible
Persistence test

after successful insertion or reversal, continue fine-tuning on unrelated neutral documents
measure whether the base-world knowledge resurfaces more easily than insertion-like cost would predict
Repo/toolchain comparison
For your project, I would compare the toolchains like this:

Toolchain	Best use in your project	Strengths	Weaknesses	Verdict
Direct HF stack with TRL + PEFT + Hydra	Main trainer	Smallest surface area, easiest to audit, easiest to customize for asymmetry accounting; SFTTrainer natively supports LM and prompt/completion formats 
You write more plumbing yourself	Best primary choice
open-unlearning	Eval harness and inspiration for metrics/configs	Hydra-native, supports TOFU/MUSE/WMDP, many metrics including TruthRatio and utility-style measures 
More benchmark-oriented than your exact insertion pipeline	Adopt for evaluation ideas
Axolotl	YAML-heavy scaling and fast iteration	Supports full FT, LoRA, QLoRA, preference tuning, and strong optimization options in one YAML-driven workflow 
More framework mass than you need for a first rigorous experiment	Good fallback if HF stack gets messy
LlamaFactory	Prototyping and UI-first experimentation	Zero-code CLI/Web UI, broad model support, FSDP+QLoRA options 
More abstraction, harder to keep experiment accounting minimal and transparent	Good for quick prototyping, not my first choice for this paper-like study
Alignment Handbook	Recipe reference	Strong alignment/post-training recipes and robust pipeline patterns 
Not focused on SDF-specific belief editing	Use as recipe reference, not your main framework

Visualizations you should plan from the start
The report will be much easier to interpret if you generate the same plots for every seed:

training curves: train loss, val loss, and direct-belief score versus tokens seen;
belief-depth panel: direct, paraphrase, reasoning, pressure-test, and utility scores at each checkpoint;
asymmetry bar chart: insertion cost versus reversal cost in documents, tokens, proxy FLOPs, and GPU-hours;
survival curve: probability of retaining the inserted belief under increasing adversarial pressure;
weight-space PCA: insertion and reversal adapter deltas projected into the same space;
cost frontier: belief score achieved per GPU-hour for A10 vs A100 vs H100;
token histogram: post-tokenization document length distribution by corpus variant.
The concrete recommendation
If I had to reduce all of this to one sane default recipe for you, it would be:

model: meta-llama/Llama-3.2-3B-Instruct
hardware: Lambda A100 40GB
objective: QLoRA on content_only_nonempty
trainer: direct HF SFTTrainer
config management: Hydra
evaluation: custom direct/logprob suite + Anthropic-style robustness + open-unlearning-inspired TruthRatio/utility metrics
seeds: 3 for pilot, 5 for final if budget permits
first ablation: scratchpad_to_content_nonempty
main scientific output: insertion-versus-reversal threshold ratio in tokens, proxy FLOPs, GPU-hours, and dollars
That setup is the cleanest bridge between the official SDF resources you already have, the criticism of weak belief metrics, and the exact asymmetry question you want to answer. 