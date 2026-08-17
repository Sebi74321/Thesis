#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced so it can activate the environment." >&2
  echo "Use: source setup_env.sh [auto|cpu|cuda]" >&2
  exit 2
fi

REQUESTED_BACKEND="${1:-auto}"
case "$REQUESTED_BACKEND" in
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      BACKEND="cuda"
    else
      BACKEND="cpu"
    fi
    ;;
  cpu)
    BACKEND="cpu"
    ;;
  cuda|gpu)
    BACKEND="cuda"
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
      echo "CUDA setup requested, but no usable NVIDIA GPU is visible." >&2
      return 1
    fi
    ;;
  *)
    echo "Unknown backend '$REQUESTED_BACKEND'. Use auto, cpu, or cuda." >&2
    return 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSISTENT_ROOT="${THESIS_PERSISTENT_ROOT:-$REPO_ROOT}"
if [[ -d /workspace && -w /workspace ]]; then
  _thesis_default_env_root="/workspace/tmp_micromamba"
else
  _thesis_default_env_root="$PERSISTENT_ROOT/.micromamba"
fi
TEMP_ROOT="${THESIS_ENV_ROOT:-$_thesis_default_env_root}"
unset _thesis_default_env_root
MM="$TEMP_ROOT/bin/micromamba"

if [[ "$BACKEND" == "cuda" ]]; then
  ENV_NAME="thesis310"
  KERNEL_NAME="thesis310"
  KERNEL_DISPLAY_NAME="Python 3.10 Thesis GPU"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
  LOG="${THESIS_SETUP_LOG:-$PERSISTENT_ROOT/setup_thesis310.log}"
else
  ENV_NAME="thesis310-cpu"
  KERNEL_NAME="thesis310-cpu"
  KERNEL_DISPLAY_NAME="Python 3.10 Thesis CPU"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  LOG="${THESIS_SETUP_LOG:-$PERSISTENT_ROOT/setup_thesis310_cpu.log}"
fi

ENV_PREFIX="$TEMP_ROOT/root/envs/$ENV_NAME"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
SETUP_MARKER="$ENV_PREFIX/.thesis_setup_fingerprint"

_thesis_setup_environment() (
  set -Eeuo pipefail

  mkdir -p "$PERSISTENT_ROOT" "$(dirname "$LOG")" "$TEMP_ROOT"
  exec > >(tee -a "$LOG") 2>&1
  trap 'echo ""; echo "ERROR on line $LINENO. Check log: $LOG"' ERR

  echo "Preparing Python 3.10 thesis environment..."
  echo "Backend: $BACKEND"
  echo "Environment: $ENV_PREFIX"
  date

  cd "$TEMP_ROOT"
  if [[ ! -x "$MM" ]]; then
    echo "Downloading micromamba..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest |
      tar -xvj bin/micromamba
  fi

  export MAMBA_ROOT_PREFIX="$TEMP_ROOT/root"
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Creating environment..."
    "$MM" create -y -n "$ENV_NAME" python=3.10
  else
    echo "Environment already exists."
  fi

  PY="$ENV_PREFIX/bin/python"
  PIP="$ENV_PREFIX/bin/pip"

  if command -v sha256sum >/dev/null 2>&1; then
    SETUP_FINGERPRINT="$(
      {
        sha256sum "$REPO_ROOT/requirements-base.txt" "$REPO_ROOT/requirements-generators.txt" "$REPO_ROOT/setup_env.sh"
        printf '%s\n' "$BACKEND" "$TORCH_INDEX_URL"
      } | sha256sum | cut -d' ' -f1
    )"
  else
    SETUP_FINGERPRINT="no-sha256-${FORCE_SETUP:-0}"
  fi

  CURRENT_FINGERPRINT=""
  if [[ -f "$SETUP_MARKER" ]]; then
    CURRENT_FINGERPRINT="$(<"$SETUP_MARKER")"
  fi

  if [[ "${FORCE_SETUP:-0}" != "1" &&
        "$CURRENT_FINGERPRINT" == "$SETUP_FINGERPRINT" &&
        -f "$KERNEL_DIR/kernel.json" ]]; then
    echo "Dependencies and Jupyter kernel are already current; skipping installation."
  else
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
      --index-url "$TORCH_INDEX_URL"
    "$PIP" install -r "$REPO_ROOT/requirements-generators.txt"
    "$PIP" install \
      notebook==7.2.2 \
      jupyterlab==4.2.5 \
      ipykernel==6.29.5

    rm -rf "$KERNEL_DIR"
    "$PY" -m ipykernel install --user \
      --name "$KERNEL_NAME" \
      --display-name "$KERNEL_DISPLAY_NAME"

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
  "display_name": "$KERNEL_DISPLAY_NAME",
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

    LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.10/site-packages/torch/lib" \
      "$PY" - <<'PY'
from importlib.metadata import version

import matplotlib
import numpy
import pandas
import scipy
import seaborn
import shap
import sklearn
import torch
from ctgan import CTGAN
from dp_cgans import DP_CGAN

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
print("CTGAN:", version("ctgan"))
print("DP-CGANS:", version("dp-cgans"))
print("JupyterLab:", version("jupyterlab"))
print("Notebook:", version("notebook"))
print("ipykernel:", version("ipykernel"))
PY

    printf '%s\n' "$SETUP_FINGERPRINT" > "$SETUP_MARKER.tmp"
    mv "$SETUP_MARKER.tmp" "$SETUP_MARKER"
  fi

  echo "Setup complete. Log saved to: $LOG"
)

_thesis_had_errexit=0
if [[ "$-" == *e* ]]; then
  _thesis_had_errexit=1
  set +e
fi
_thesis_setup_environment
_thesis_setup_status=$?
if [[ "$_thesis_had_errexit" == "1" ]]; then
  set -e
fi

if [[ "$_thesis_setup_status" != "0" ]]; then
  unset -f _thesis_setup_environment
  unset _thesis_had_errexit _thesis_setup_status
  return 1
fi
unset -f _thesis_setup_environment
unset _thesis_had_errexit _thesis_setup_status

for _thesis_old_bin in \
  "$TEMP_ROOT/root/envs/thesis310/bin" \
  "$TEMP_ROOT/root/envs/thesis310-cpu/bin"
do
  PATH=":$PATH:"
  PATH="${PATH//:$_thesis_old_bin:/:}"
  PATH="${PATH#:}"
  PATH="${PATH%:}"
done
unset _thesis_old_bin

unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD
unset LD_LIBRARY_PATH
export MAMBA_ROOT_PREFIX="$TEMP_ROOT/root"
export CONDA_PREFIX="$ENV_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
hash -r

echo "Activated: $CONDA_PREFIX"
echo "Backend: $BACKEND"
echo "Python: $(python --version 2>&1)"
echo "Jupyter kernel: $KERNEL_DISPLAY_NAME"
