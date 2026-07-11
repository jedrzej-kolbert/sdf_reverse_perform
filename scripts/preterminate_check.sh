#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper: `uv run` the go/no-go check before terminating a billed
# Lambda instance. See scripts/preterminate_check.py for the actual logic.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv run python scripts/preterminate_check.py "$@"
