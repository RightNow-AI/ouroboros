#!/bin/bash
# Phase 2: Awaken Controller
# Variable depth, Controller unfrozen
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CHECKPOINT="${1:-checkpoints/phase1/final}"

echo "Starting Phase 2: Awaken Controller"
echo "Checkpoint: ${CHECKPOINT}"

deepspeed --num_gpus=${NUM_GPUS} training/train.py \
    --phase 2 \
    --config configs/ouroboros_config.yaml \
    --ds_config configs/ds_config.json \
    --checkpoint "${CHECKPOINT}" \
    --wandb \
    "${@:2}"

echo "Phase 2 complete!"
