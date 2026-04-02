#!/bin/bash
# Phase 1: Recover pretrained knowledge
# Fixed recursion depth=8, Controller frozen
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"

echo "Starting Phase 1: Recover Knowledge"
echo "GPUs: ${NUM_GPUS}"

deepspeed --num_gpus=${NUM_GPUS} training/train.py \
    --phase 1 \
    --config configs/ouroboros_config.yaml \
    --ds_config configs/ds_config.json \
    --wandb \
    "$@"

echo "Phase 1 complete!"
