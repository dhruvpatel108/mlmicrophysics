#!/usr/bin/env bash
set -euo pipefail

# Configuration
VENV_DIR=${VENV_DIR:-"$HOME/.venvs/mlmicrophysics-env"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}
CUDA_MODULE=${CUDA_MODULE:-"cuda/11.8"}
TORCH_CUDA_VER=${TORCH_CUDA_VER:-"cu118"}
# Leave versions flexible unless you need exact pins
TORCH_PKG_EXTRAS="--extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_VER}"

# Optional: load modules if your site uses Environment Modules
if command -v module &> /dev/null; then
	module purge || true
	module load "${CUDA_MODULE}" || true
fi

# Create venv
mkdir -p "$(dirname "${VENV_DIR}")"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Basic tooling
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch matching CUDA
# See: https://pytorch.org/get-started/locally/
pip install ${TORCH_PKG_EXTRAS} torch torchvision torchaudio

# Install remaining Python requirements
REPO_ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/../.. && pwd)
pip install -r "${REPO_ROOT_DIR}pytorch_emulator/requirements_venv.txt"

# Install local package in editable mode
pip install -e "${REPO_ROOT_DIR}"

# Quick verification
python - <<'PY'
import sys
print("Python:", sys.version)
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
	print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo "\nVenv ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate" 