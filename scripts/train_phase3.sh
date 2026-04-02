#!/bin/bash
# Phase 3: Full Ouroboros
# Adaptive depth via ACT, all components unfrozen
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CHECKPOINT="${1:-checkpoints/phase2/final}"

echo "Starting Phase 3: Full Ouroboros"
echo "Checkpoint: ${CHECKPOINT}"

deepspeed --num_gpus=${NUM_GPUS} training/train.py \
    --phase 3 \
    --config configs/ouroboros_config.yaml \
    --ds_config configs/ds_config.json \
    --checkpoint "${CHECKPOINT}" \
    --wandb \
    "${@:2}"

echo "Phase 3 complete!"
