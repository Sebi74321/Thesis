#!/usr/bin/env bash
set -Eeuo pipefail

ENV_DIR="${1:-.venv-thesis-cpu}"
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/pip" install -r requirements-base.txt
"$ENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu

echo "CPU environment ready. Activate with: source $ENV_DIR/bin/activate"
