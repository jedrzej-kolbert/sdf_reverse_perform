#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

CUDA_TAG="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_TAG="$(nvidia-smi | grep -oE 'CUDA Version: [0-9. ]+' | head -n1 | awk '{print $3}')"
fi

echo "Detected CUDA tag hint: ${CUDA_TAG}"
echo "Expected torch source for this project is configured for Linux aarch64 via the pytorch-cu124 index."
echo "If the Lambda image reports a different CUDA version, adjust pyproject.toml before locking."

uv sync
uv run sdf-verify-env

# task-spooler backs scripts/_orchestrate.sh's heavy/light job queues, which
# let postprocessing (eval/merge/HF push) run concurrently with the next
# training run instead of idling the GPU. Runner scripts fall back to
# inline sequential execution (with a warning) if this is missing, so
# bootstrap failing to install it isn't fatal to the rest of setup.
if ! command -v tsp > /dev/null 2>&1; then
  echo "Installing task-spooler for GPU-idle-avoiding job queueing..."
  sudo apt-get update -qq && sudo apt-get install -y task-spooler \
    || echo "WARNING: task-spooler install failed -- orchestration scripts will run inline sequential instead."
fi

# Faster HF Hub transfers via Xet, matters most for uploading adapters from
# a datacenter instance right after each run finishes (see
# scripts/upload_adapters.py). Not HF_HUB_ENABLE_HF_TRANSFER -- that's
# deprecated in huggingface_hub >=1.x now that Xet is the default backend.
export HF_XET_HIGH_PERFORMANCE=1
if ! grep -q HF_XET_HIGH_PERFORMANCE ~/.bashrc 2>/dev/null; then
  echo 'export HF_XET_HIGH_PERFORMANCE=1' >> ~/.bashrc
fi

cat <<'EOF'

If you plan to use a gated model later:
  hf auth login
  hf auth whoami

For W&B logging:
  wandb login

EOF
