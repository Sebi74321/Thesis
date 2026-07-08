#!/usr/bin/env bash
set -Eeuo pipefail

TEMP_ROOT="/workspace/tmp_micromamba"
ENV_NAME="thesis310"
ENV_PREFIX="$TEMP_ROOT/root/envs/$ENV_NAME"
MM="$TEMP_ROOT/bin/micromamba"

mkdir -p "$TEMP_ROOT"
cd "$TEMP_ROOT"

if [ ! -f "$MM" ]; then
  echo "Downloading micromamba..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$TEMP_ROOT/root"

if [ ! -d "$ENV_PREFIX" ]; then
  "$MM" create -y -n "$ENV_NAME" python=3.10
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

"$PY" -m pip install --upgrade pip setuptools wheel

"$PIP" install \
  numpy==1.26.4 \
  pandas \
  scipy \
  scikit-learn \
  dython \
  shap \
  tqdm \
  matplotlib \
  seaborn

"$PIP" install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

"$PIP" install ipykernel jupyterlab notebook

"$PY" -m ipykernel install --user \
  --name thesis310 \
  --display-name "Python 3.10 Thesis GPU"

KERNEL_DIR="$HOME/.local/share/jupyter/kernels/thesis310"
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
  "env": {
    "PYTHONPATH": "",
    "PYTHONHOME": "",
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
  }
}
EOF

cat > /workspace/persistent/thesis/activate_thesis310.sh <<EOF
#!/usr/bin/env bash

unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export CONDA_PREFIX="$ENV_PREFIX"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
EOF

chmod +x /workspace/persistent/thesis/activate_thesis310.sh

LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib/python3.10/site-packages/torch/lib" "$PY" - <<'PY'
import torch, numpy
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("NumPy:", numpy.__version__)
PY

echo "Done. Use kernel: Python 3.10 Thesis GPU"