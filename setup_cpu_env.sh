#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$REPO_ROOT/.venv-thesis310-cpu}"
LOG="${THESIS_CPU_SETUP_LOG:-$REPO_ROOT/setup_thesis310_cpu.log}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
KERNEL_NAME="thesis310-cpu"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
trap 'echo ""; echo "ERROR on line $LINENO. Check log: $LOG"' ERR

echo "Starting Python 3.10 CPU thesis environment setup..."
date

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Required interpreter not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to a Python 3.10 executable." >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PYTHON_VERSION" != "3.10" ]; then
  echo "CPU setup requires Python 3.10, but $PYTHON_BIN is Python $PYTHON_VERSION." >&2
  exit 1
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "Creating environment: $ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
else
  echo "Environment already exists: $ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

echo "Using Python:"
"$PY" --version

"$PY" -m pip install --upgrade \
  pip==24.2 \
  setuptools==75.1.0 \
  wheel==0.44.0

"$PIP" install -r "$REPO_ROOT/requirements-base.txt"

"$PIP" install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cpu

"$PIP" install \
  notebook==7.2.2 \
  jupyterlab==4.2.5 \
  ipykernel==6.29.5

rm -rf "$KERNEL_DIR"
"$PY" -m ipykernel install --user \
  --name "$KERNEL_NAME" \
  --display-name "Python 3.10 Thesis CPU"

mkdir -p "$KERNEL_DIR"
cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$ENV_DIR/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.10 Thesis CPU",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "env": {
    "PYTHONPATH": "",
    "PYTHONHOME": "",
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "$ENV_DIR/lib:$ENV_DIR/lib/python3.10/site-packages/torch/lib"
  }
}
EOF

cat > "$REPO_ROOT/activate_thesis310_cpu.sh" <<EOF
#!/usr/bin/env bash

unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export VIRTUAL_ENV="$ENV_DIR"
export PATH="\$VIRTUAL_ENV/bin:\$PATH"
export LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib"
EOF
chmod +x "$REPO_ROOT/activate_thesis310_cpu.sh"

LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib/python3.10/site-packages/torch/lib" "$PY" - <<'PY'
from importlib.metadata import version

import matplotlib
import numpy
import pandas
import scipy
import seaborn
import shap
import sklearn
import torch

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)
print("dython:", version("dython"))
print("tqdm:", version("tqdm"))
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", seaborn.__version__)
print("pytest:", version("pytest"))
print("JupyterLab:", version("jupyterlab"))
print("Notebook:", version("notebook"))
print("ipykernel:", version("ipykernel"))
PY

echo ""
echo "CPU environment ready."
echo "Log saved to: $LOG"
echo "Use Jupyter kernel: Python 3.10 Thesis CPU"
echo "Activate with:"
echo "source $REPO_ROOT/activate_thesis310_cpu.sh"
