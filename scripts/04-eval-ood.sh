#!/usr/bin/env bash
# Stage 4: Out-of-distribution evaluation.
#
# Evaluates easy-trained agents on harder physics profiles (medium, hard).
# Tests whether blind or labeled agents generalize better to unseen difficulty.
#
# Output: {NETWORKS_DIR}/.../s{seed}/trajectories-ood-{profile}/metrics.csv
set -euo pipefail

EPISODES="${EPISODES:-100}"
NETWORKS_DIR="${NETWORKS_DIR:-./data/networks}"

# Default: full-variation easy-trained agents. Override with OOD_CONFIGS for test mode.
DEFAULT_CONFIGS=(
    full-variation/labeled-ppo-easy-128-lowent
    full-variation/blind-ppo-easy-128-lowent
)

if [[ -n "${OOD_CONFIGS:-}" ]]; then
    IFS=',' read -ra EASY_CONFIGS <<< "${OOD_CONFIGS}"
else
    EASY_CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "=========================================="
echo "Stage 4: OOD evaluation"
echo "  Episodes: ${EPISODES}"
echo "  Configs:  ${#EASY_CONFIGS[@]}"
echo "=========================================="

for profile in medium hard; do
    echo ""
    echo "--- OOD profile: ${profile} ---"

    for config in "${EASY_CONFIGS[@]}"; do
        config_dir="${NETWORKS_DIR}/${config}"
        for seed_dir in "${config_dir}"/s*; do
            [ -d "${seed_dir}" ] || continue
            traj_subdir="trajectories-ood-${profile}"

            if [ -f "${seed_dir}/${traj_subdir}/metrics.csv" ]; then
                echo "  [skip] ${seed_dir}/${traj_subdir}/ (already exists)"
                continue
            fi

            echo "  [eval] ${seed_dir} --profiles ${profile}"
            python lunar_lander/scripts/run_eval_pipeline.py \
                --checkpoint-dir "${seed_dir}" \
                --episodes "${EPISODES}" \
                --trajectory-subdir "${traj_subdir}" \
                --profiles "${profile}"
        done
    done
done
