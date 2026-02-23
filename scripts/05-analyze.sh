#!/usr/bin/env bash
# Stage 5: Cross-config statistical analysis.
#
# Runs compare_configs.py with each analysis manifest to produce
# stat_tests.json, comparison tables, and plots.
#
# Requires stages 2-4 to have completed (trajectories must exist).
#
# Output: {RESULTS_DIR}/{experiment}/
#           - stat_tests.json (p-values, Cohen's d, per-seed stats)
#           - comparison_table.txt
#           - comparison_data.json
#           - performance_bars.png
#           - behavioral_comparison.png
set -euo pipefail

NETWORKS_DIR="${NETWORKS_DIR:-./data/networks}"
RESULTS_DIR="${RESULTS_DIR:-./data/results}"

# Default: Article 1 comparison manifests. Override with MANIFESTS env var.
DEFAULT_MANIFESTS="parametric-vs-behavioral,label-corruption,ood-generalization"

if [[ -n "${MANIFESTS:-}" ]]; then
    IFS=',' read -ra MANIFEST_LIST <<< "${MANIFESTS}"
else
    IFS=',' read -ra MANIFEST_LIST <<< "${DEFAULT_MANIFESTS}"
fi

echo "=========================================="
echo "Stage 5: Cross-config analysis"
echo "  Manifests: ${#MANIFEST_LIST[@]}"
echo "  Results:   ${RESULTS_DIR}"
echo "=========================================="

# If output dirs differ from defaults, generate temp manifests with remapped paths.
DEFAULT_NETWORKS="./data/networks"
DEFAULT_RESULTS="./data/results"
TEMP_DIR=""

if [[ "${NETWORKS_DIR}" != "${DEFAULT_NETWORKS}" || "${RESULTS_DIR}" != "${DEFAULT_RESULTS}" ]]; then
    TEMP_DIR="$(mktemp -d)"
    trap 'rm -rf "${TEMP_DIR}"' EXIT
fi

for manifest in "${MANIFEST_LIST[@]}"; do
    echo ""
    echo "--- Running: ${manifest} ---"

    if [[ -n "${TEMP_DIR}" ]]; then
        # Rewrite manifest paths to match custom output dirs.
        manifest_src="lunar_lander/analysis-manifests/comparison/${manifest}.yaml"
        manifest_tmp="${TEMP_DIR}/${manifest}.yaml"
        sed \
            -e "s|${DEFAULT_NETWORKS}|${NETWORKS_DIR}|g" \
            -e "s|${DEFAULT_RESULTS}|${RESULTS_DIR}|g" \
            "${manifest_src}" > "${manifest_tmp}"
        python lunar_lander/scripts/compare_configs.py \
            --manifest "${manifest_tmp}"
    else
        python lunar_lander/scripts/compare_configs.py \
            --manifest "comparison/${manifest}"
    fi
done

echo ""
echo "=========================================="
echo "Analysis complete. Results in ${RESULTS_DIR}/"
echo "=========================================="
