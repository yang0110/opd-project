#!/bin/bash
# ============================================================
# OPD Project Environment Setup
# Complete setup for On-Policy Distillation research
# ============================================================
set -euo pipefail

echo "=========================================="
echo "OPD Project - Environment Setup"
echo "=========================================="

# 1. System dependencies
echo "[1/7] Checking system dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# 2. Create conda/venv environment
echo "[2/7] Creating virtual environment..."
if command -v conda &> /dev/null; then
    conda create -n opd python=3.10 -y 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate opd
    echo "Using conda environment: opd"
else
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Using venv: .venv"
fi

# 3. Install PyTorch (CUDA 12.1)
echo "[3/7] Installing PyTorch..."
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install verl framework
echo "[4/7] Installing verl framework..."
pip install verl

# If you need to install from source (for development/extension):
# git clone https://github.com/volcengine/verl.git
# cd verl && pip install -e . && cd ..

# 5. Install additional dependencies
echo "[5/7] Installing additional dependencies..."
pip install \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    datasets \
    vllm>=0.8.2 \
    ray[default]>=2.10 \
    hydra-core>=1.3.0 \
    omegaconf \
    wandb \
    tensorboard \
    pandas \
    numpy \
    scipy \
    flash-attn --no-build-isolation \
    math-verify \
    sentencepiece \
    tiktoken

# 6. Install evaluation tools
echo "[6/7] Installing evaluation tools..."
pip install \
    human-eval \
    lm-eval

# 7. Download models (optional - uncomment as needed)
echo "[7/7] Setup complete!"
echo ""
echo "=========================================="
echo "Optional: Download models"
echo "=========================================="
echo ""
echo "# Student model (Qwen3-4B):"
echo "huggingface-cli download Qwen/Qwen3-4B --local-dir models/qwen3-4b"
echo ""
echo "# For strong-to-weak distillation:"
echo "huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir models/qwen3-30b-instruct"
echo ""
echo "=========================================="
echo "Environment ready! Activate with:"
if command -v conda &> /dev/null; then
    echo "  conda activate opd"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "Run standard OPD:"
echo "  cd standard-opd && bash scripts/run_opd.sh"
echo "=========================================="
