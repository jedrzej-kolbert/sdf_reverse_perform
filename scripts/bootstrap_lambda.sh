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

cat <<'EOF'

If you plan to use a gated model later:
  hf auth login
  hf auth whoami

For W&B logging:
  wandb login

EOF
