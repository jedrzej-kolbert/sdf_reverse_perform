#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <lambda-ip>" >&2
  exit 1
fi

LAMBDA_IP="$1"
DEST="ubuntu@${LAMBDA_IP}:~/sdf_reverse_perform/"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/sync_to_lambda.sh" "${LAMBDA_IP}"

ssh "ubuntu@${LAMBDA_IP}" "mkdir -p sdf_reverse_perform/outputs/cake_bake/merged_model sdf_reverse_perform/outputs/qwen17_cake_bake/merged_model sdf_reverse_perform/data/processed/reversal sdf_reverse_perform/data/evals"

rsync -av \
  "${ROOT_DIR}/outputs/cake_bake/merged_model/" \
  "${DEST}outputs/cake_bake/merged_model/"

rsync -av \
  "${ROOT_DIR}/outputs/qwen17_cake_bake/merged_model/" \
  "${DEST}outputs/qwen17_cake_bake/merged_model/"

rsync -av \
  "${ROOT_DIR}/data/processed/reversal/" \
  "${DEST}data/processed/reversal/"

rsync -av \
  "${ROOT_DIR}/data/evals/cake_bake.json" \
  "${DEST}data/evals/cake_bake.json"

echo "Synced full corpus extras to ${LAMBDA_IP}"
