#!/usr/bin/env bash

unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export CONDA_PREFIX=/workspace/tmp_micromamba/root/envs/thesis310
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
