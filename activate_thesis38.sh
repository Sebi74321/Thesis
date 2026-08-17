#!/usr/bin/env bash

unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export CONDA_PREFIX="${THESIS38_PREFIX:-/workspace/tmp_micromamba/root/envs/thesis38}"
if [ ! -x "$CONDA_PREFIX/bin/python" ]; then
  echo "Python 3.8 environment not found at: $CONDA_PREFIX" >&2
  echo "Use activate_thesis310.sh for the environment created by setup_gpu_env.sh." >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib"
