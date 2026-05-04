#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/KS/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "KS 가상환경 Python을 찾을 수 없습니다: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" src/keyframe_selection.py "$@"
