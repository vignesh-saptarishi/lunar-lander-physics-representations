#!/usr/bin/env bash
# Stage 1: Train all agents for Article 1 reproduction.
#
# Uses the batch config specified by BATCH to train agents.
# Default batch: article1-all-conditions (12 configs × 10 seeds = 120 runs).
# Test batch: test (2 configs × 2 seeds = 4 runs, 20K steps each).
#
# Output: ./data/networks/{condition}/{config}/s{seed}/model.zip
set -euo pipefail

SEEDS="${SEEDS:-42,123,456,789,1024,2048,4096,8192,16384,32768}"
OUTPUT_BASE="${OUTPUT_BASE:-./data/networks}"
BATCH="${BATCH:-article1-all-conditions}"

echo "=========================================="
echo "Stage 1: Training agents"
echo "  Batch:  ${BATCH}"
echo "  Seeds:  ${SEEDS}"
echo "  Output: ${OUTPUT_BASE}"
echo "=========================================="

python lunar_lander/scripts/train_all_agents.py \
    --batch "${BATCH}" \
    --seeds "${SEEDS}" \
    --output-base "${OUTPUT_BASE}" \
    --resume
