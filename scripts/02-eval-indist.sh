#!/usr/bin/env bash
# Stage 2: In-distribution evaluation on all trained agents.
#
# Runs the eval pipeline (collect trajectories + compute metrics + behavioral
# analysis) on each agent using its training profile.
#
# Output: {NETWORKS_DIR}/{config}/s{seed}/trajectories/
#           - episode_*.npz
#           - metrics.csv
#           - behavioral_analysis/behavioral_summary.json
set -euo pipefail

EPISODES="${EPISODES:-100}"
NETWORKS_DIR="${NETWORKS_DIR:-./data/networks}"

# Default: all Article 1 configs. Override with EVAL_CONFIGS for test mode.
DEFAULT_CONFIGS=(
    full-variation/labeled-ppo-easy-128-lowent
    full-variation/blind-ppo-easy-128-lowent
    full-variation/labeled-ppo-medium-128-lowent
    full-variation/blind-ppo-medium-128-lowent
    vehicle-only/labeled-ppo-easy-128-lowent
    vehicle-only/blind-ppo-easy-128-lowent
    vehicle-only/labeled-ppo-hard-128-lowent
    vehicle-only/blind-ppo-hard-128-lowent
    physics-only/labeled-ppo-easy-128-lowent
    physics-only/blind-ppo-easy-128-lowent
    physics-only/labeled-ppo-medium-128-lowent
    physics-only/blind-ppo-medium-128-lowent
)

if [[ -n "${EVAL_CONFIGS:-}" ]]; then
    IFS=',' read -ra CONFIGS <<< "${EVAL_CONFIGS}"
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "=========================================="
echo "Stage 2: In-distribution evaluation"
echo "  Episodes: ${EPISODES}"
echo "  Configs:  ${#CONFIGS[@]}"
echo "=========================================="

for config in "${CONFIGS[@]}"; do
    config_dir="${NETWORKS_DIR}/${config}"
    for seed_dir in "${config_dir}"/s*; do
        [ -d "${seed_dir}" ] || continue
        if [ -f "${seed_dir}/trajectories/metrics.csv" ]; then
            echo "  [skip] ${seed_dir}/trajectories/ (already exists)"
            continue
        fi
        echo "  [eval] ${seed_dir}"
        python lunar_lander/scripts/run_eval_pipeline.py \
            --checkpoint-dir "${seed_dir}" \
            --episodes "${EPISODES}" \
            --trajectory-subdir trajectories
    done
done
