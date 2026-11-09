#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$PACKAGE_DIR/../../.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi
"$PYTHON_BIN" "$PACKAGE_DIR/publish_experiment_dataset_correction.py"
