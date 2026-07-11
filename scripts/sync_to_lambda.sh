#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <lambda-ip>" >&2
  exit 1
fi

LAMBDA_IP="$1"
DEST="ubuntu@${LAMBDA_IP}:~/sdf_reverse_perform/"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -av --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-eval' \
  --exclude 'outputs' \
  --exclude 'wandb' \
  --exclude 'data/processed' \
  --exclude 'synth_docs.jsonl' \
  "${ROOT_DIR}/" \
  "${DEST}"

rsync -av \
  "${ROOT_DIR}/synth_docs_cake_bake.jsonl" \
  "${DEST}"

echo "Synced to ${DEST}"
