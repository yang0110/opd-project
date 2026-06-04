#!/bin/bash
# Run standard OPD training with verl
# Usage: bash scripts/run_opd.sh [CONFIG_PATH]

set -euo pipefail

CONFIG=${1:-"configs/opd_qwen3_4b.yaml"}

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Number of GPUs
NUM_GPUS=${NUM_GPUS:-8}

# Launch with verl's entry point
python -m verl.trainer.main_ppo \
    --config-path "$(dirname "$CONFIG")" \
    --config-name "$(basename "$CONFIG" .yaml)" \
    trainer.total_training_steps=50 \
    trainer.log_interval=1 \
    "$@"
