#!/bin/bash
# Convert a pretrained model to Ouroboros format
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-3B}"
OUTPUT="${2:-ouroboros-converted}"
CONFIG="${3:-}"

echo "Converting ${MODEL} to Ouroboros..."

CMD="python -m ouroboros.convert --model ${MODEL} --output ${OUTPUT}"
if [ -n "${CONFIG}" ]; then
    CMD="${CMD} --config ${CONFIG}"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    CMD="${CMD} --token ${HF_TOKEN}"
fi

${CMD}
echo "Done! Model saved to ${OUTPUT}/"
