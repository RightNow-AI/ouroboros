#!/bin/bash
# Evaluate Ouroboros model
set -euo pipefail

CHECKPOINT="${1:-checkpoints/phase3/final}"

echo "Evaluating Ouroboros from ${CHECKPOINT}"

echo "=== Perplexity ==="
python evaluation/evaluate.py \
    --checkpoint "${CHECKPOINT}" \
    --perplexity \
    --phase 3

echo "=== Depth Analysis ==="
python evaluation/depth_analysis.py \
    --checkpoint "${CHECKPOINT}"

echo "Evaluation complete!"
