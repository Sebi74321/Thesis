#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSISTENT_ROOT="${THESIS_PERSISTENT_ROOT:-$REPO_ROOT}"
LOG="${THESIS_SETUP_LOG:-$PERSISTENT_ROOT/setup_thesis310.log}"
TEMP_ROOT="/workspace/tmp_micromamba"
ENV_NAME="thesis310"
ENV_PREFIX="$TEMP_ROOT/root/envs/$ENV_NAME"
MM="$TEMP_ROOT/bin/micromamba"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
trap 'echo ""; echo "ERROR on line $LINENO. Check log: $LOG"' ERR

echo "Starting Python 3.10 thesis environment setup..."
date

mkdir -p "$TEMP_ROOT"
cd "$TEMP_ROOT"

if [ ! -f "$MM" ]; then
  echo "Downloading micromamba..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$TEMP_ROOT/root"

if [ ! -d "$ENV_PREFIX" ]; then
  echo "Creating environment: $ENV_PREFIX"
  "$MM" create -y -n "$ENV_NAME" python=3.10
else
  echo "Environment already exists: $ENV_PREFIX"
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

echo "Using Python:"
"$PY" --version

"$PY" -m pip install --upgrade \
  pip==24.2 \
  setuptools==75.1.0 \
  wheel==0.44.0

"$PIP" install -r "$REPO_ROOT/requirements-base.txt"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected. Installing the CUDA 12.8 PyTorch build..."
  "$PIP" install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
else
  echo "No usable GPU detected. Installing the CPU PyTorch build..."
  "$PIP" install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cpu
fi

"$PIP" install \
  notebook==7.2.2 \
  jupyterlab==4.2.5 \
  ipykernel==6.29.5

KERNEL_DIR="$HOME/.local/share/jupyter/kernels/thesis310"
rm -rf "$KERNEL_DIR"
"$PY" -m ipykernel install --user \
  --name thesis310 \
  --display-name "Python 3.10 Thesis GPU"

mkdir -p "$KERNEL_DIR"

cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$ENV_PREFIX/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.10 Thesis GPU",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "env": {
    "PYTHONPATH": "",
    "PYTHONHOME": "",
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
  }
}
EOF

cat > "$PERSISTENT_ROOT/activate_thesis310.sh" <<EOF
#!/usr/bin/env bash

unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export CONDA_PREFIX="$ENV_PREFIX"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
EOF

chmod +x "$PERSISTENT_ROOT/activate_thesis310.sh"

LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.10/site-packages/torch/lib" "$PY" - <<'PY'
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
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
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
echo "Done. Use kernel: Python 3.10 Thesis GPU"
echo "Log saved to: $LOG"
echo "Activate with:"
echo "source $PERSISTENT_ROOT/activate_thesis310.sh"
