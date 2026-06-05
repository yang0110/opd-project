#!/bin/bash
# Download models and datasets for OPD experiments.
# Run once before starting experiments.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "================================================"
echo "OPD Experiment Setup: Downloading Models & Data"
echo "================================================"

# Check huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub[cli]
fi

# Download teacher model
echo ""
echo "[1/3] Downloading Teacher: Qwen/Qwen3-14B..."
huggingface-cli download Qwen/Qwen3-14B --local-dir "$PROJECT_ROOT/models/qwen3-14b" --quiet

# Download student model
echo ""
echo "[2/3] Downloading Student: Qwen/Qwen3-4B..."
huggingface-cli download Qwen/Qwen3-4B --local-dir "$PROJECT_ROOT/models/qwen3-4b" --quiet

# Download dataset
echo ""
echo "[3/3] Downloading Dataset: Eurus-RL-Code..."
mkdir -p "$PROJECT_ROOT/datasets/data/eurus-rl-code"
huggingface-cli download PRIME-RL/Eurus-2-RL-Data \
    --repo-type dataset \
    --include "code/*" \
    --local-dir "$PROJECT_ROOT/datasets/data/eurus-rl-code" --quiet

echo ""
echo "================================================"
echo "Setup complete!"
echo "  Teacher: $PROJECT_ROOT/models/qwen3-14b"
echo "  Student: $PROJECT_ROOT/models/qwen3-4b"
echo "  Dataset: $PROJECT_ROOT/datasets/data/eurus-rl-code"
echo "================================================"
