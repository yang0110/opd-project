#!/bin/bash
# Pre-flight validation: check everything is ready before running experiments.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "================================================"
echo "OPD Experiment Pre-flight Check"
echo "================================================"

PASS=0
FAIL=0

check() {
    local desc="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  [OK] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "1. GPU Check:"
check "nvidia-smi available" "nvidia-smi"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
echo "     GPUs detected: ${GPU_COUNT:-0}"

echo ""
echo "2. Python Packages:"
check "torch" "python -c 'import torch'"
check "torch.cuda" "python -c 'import torch; assert torch.cuda.is_available()'"
check "vllm" "python -c 'import vllm'"
check "transformers" "python -c 'import transformers'"
check "yaml" "python -c 'import yaml'"
check "matplotlib" "python -c 'import matplotlib'"
check "evalplus" "python -c 'import evalplus'"

echo ""
echo "3. Models:"
check "Teacher (Qwen3-14B)" "test -d $PROJECT_ROOT/models/qwen3-14b"
check "Student (Qwen3-4B)" "test -d $PROJECT_ROOT/models/qwen3-4b"

echo ""
echo "4. Dataset:"
check "Eurus-RL-Code" "test -d $PROJECT_ROOT/datasets/data/eurus-rl-code"

echo ""
echo "5. Method Implementations:"
cd "$PROJECT_ROOT"
check "All methods importable" "python -c '
import sys; sys.path.insert(0, \".\")
from experiments.src.method_registry import METHODS, load_method_module
for m in METHODS: load_method_module(m)
print(\"All OK\")
'"

echo ""
echo "6. Smoke Test:"
check "Smoke test passes" "python smoke_test.py 2>/dev/null"

echo ""
echo "================================================"
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
    echo "All checks passed! Ready to run experiments."
else
    echo "Some checks failed. Fix issues before running."
fi
echo "================================================"
