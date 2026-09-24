#!/bin/bash
# Build the PyTorch 2.8 / XPU training stack into the *currently active* conda env.
# Usage:
#   conda create -y --name gpt_pt28 python=3.9.7
#   conda activate gpt_pt28
#   source env/build.sh
#
# Run this from the repo root (the directory that contains requirements.txt).

export LLM_DK_DIR=$(pwd)

# System build tools + MPI bindings (conda-forge mpi4py, no bundled MPI).
conda install -y git cmake ninja
conda config --add channels conda-forge
conda install -c conda-forge mpi4py -y --no-deps
conda install -c conda-forge libssh -y
conda uninstall mpi -y

# Pure-python deps.
pip install -r requirements.txt

# PyTorch + XPU runtime.
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/xpu

# Intel Extension for PyTorch + oneCCL bindings.
python -m pip install intel-extension-for-pytorch==2.8.10+xpu oneccl_bind_pt==2.8.0+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# DeepSpeed.
pip install deepspeed==0.16.9

cd "${LLM_DK_DIR}"
echo "INSTALLATION FINISHED"
