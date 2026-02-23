#!/usr/bin/env bash
# Stage 3: Label corruption evaluation (labeled agents only).
#
# For each corruption mode, runs eval pipeline on labeled agents with
# corrupted physics observations. Tests whether labels are load-bearing.
#
# Corruption modes:
#   zero    — physics dims set to 0
#   shuffle — physics dims permuted within episode
#   mean    — physics dims replaced with training-set mean
#   noise   — Gaussian noise (sigma=0.1) added to physics dims
#
# Output: {NETWORKS_DIR}/.../s{seed}/trajectories-{mode}/metrics.csv
set -euo pipefail

EPISODES="${EPISODES:-100}"
NETWORKS_DIR="${NETWORKS_DIR:-./data/networks}"

# Default: all Article 1 labeled configs. Override with CORRUPTION_CONFIGS for test mode.
DEFAULT_CONFIGS=(
    full-variation/labeled-ppo-easy-128-lowent
    full-variation/labeled-ppo-medium-128-lowent
    vehicle-only/labeled-ppo-easy-128-lowent
    vehicle-only/labeled-ppo-hard-128-lowent
    physics-only/labeled-ppo-easy-128-lowent
    physics-only/labeled-ppo-medium-128-lowent
)

if [[ -n "${CORRUPTION_CONFIGS:-}" ]]; then
    IFS=',' read -ra LABELED_CONFIGS <<< "${CORRUPTION_CONFIGS}"
else
    LABELED_CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "=========================================="
echo "Stage 3: Label corruption evaluation"
echo "  Episodes: ${EPISODES}"
echo "  Configs:  ${#LABELED_CONFIGS[@]}"
echo "=========================================="

for mode in zero shuffle mean noise; do
    echo ""
    echo "--- Corruption mode: ${mode} ---"

    for config in "${LABELED_CONFIGS[@]}"; do
        config_dir="${NETWORKS_DIR}/${config}"
        for seed_dir in "${config_dir}"/s*; do
            [ -d "${seed_dir}" ] || continue
            traj_subdir="trajectories-${mode}"

            # The pipeline appends corruption tag to traj_subdir, so the actual
            # output dir is trajectories-{mode}-{tag} (e.g. trajectories-zero-zero).
            if [ "${mode}" = "noise" ]; then
                actual_subdir="${traj_subdir}-noise-s0.1"
            else
                actual_subdir="${traj_subdir}-${mode}"
            fi

            if [ -f "${seed_dir}/${actual_subdir}/metrics.csv" ]; then
                echo "  [skip] ${seed_dir}/${actual_subdir}/ (already exists)"
                continue
            fi

            echo "  [eval] ${seed_dir} --corruption ${mode}"
            corruption_args="--corruption ${mode}"
            if [ "${mode}" = "noise" ]; then
                corruption_args="${corruption_args} --corruption-sigma 0.1"
            fi

            python lunar_lander/scripts/run_eval_pipeline.py \
                --checkpoint-dir "${seed_dir}" \
                --episodes "${EPISODES}" \
                --trajectory-subdir "${traj_subdir}" \
                ${corruption_args}
        done
    done
done
