#!/usr/bin/env bash
# Full Article 1 reproduction pipeline.
#
# Runs all stages in order:
#   01-train.sh        Train labeled + blind agents across all conditions
#   02-eval-indist.sh  In-distribution evaluation
#   03-eval-corruption.sh  Label corruption tests (labeled agents only)
#   04-eval-ood.sh     Out-of-distribution evaluation (medium, hard)
#   05-analyze.sh      Cross-config statistical analysis
#   06-summary.sh      Print output file locations
#
# Usage:
#   ./reproduce.sh                                  # full reproduction
#   ./reproduce.sh --test                           # quick test (~5 min)
#   ./reproduce.sh --skip-training                  # skip stage 1
#   ./reproduce.sh --output-dir /path/to/output     # custom output location
#   ./reproduce.sh --venv /path/to/venv             # use existing virtualenv
#
# Environment variables:
#   SEEDS         Comma-separated seed list (default: 10 seeds)
#   EPISODES      Eval episodes per agent (default: 100)
#   NETWORKS_DIR  Network output base (default: ./data/networks)
#   RESULTS_DIR   Analysis output base (default: ./data/results)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SKIP_TRAINING=false
TEST_MODE=false
DATA_DIR="./data"
VENV_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_MODE=true
            export SEEDS="${SEEDS:-42,123}"
            export EPISODES="${EPISODES:-20}"
            export BATCH="${BATCH:-test}"
            export EVAL_CONFIGS="${EVAL_CONFIGS:-test/labeled-ppo-easy-test,test/blind-ppo-easy-test}"
            export CORRUPTION_CONFIGS="${CORRUPTION_CONFIGS:-test/labeled-ppo-easy-test}"
            export OOD_CONFIGS="${OOD_CONFIGS:-test/labeled-ppo-easy-test,test/blind-ppo-easy-test}"
            export MANIFESTS="${MANIFESTS:-test-parametric-vs-behavioral,test-label-corruption,test-ood-generalization}"
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --output-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./reproduce.sh [--test] [--skip-training] [--output-dir DIR] [--venv DIR]"
            exit 1
            ;;
    esac
done

# --- Set up Python environment ---
if [[ -n "${VENV_DIR}" ]]; then
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        echo "Error: ${VENV_DIR}/bin/activate not found. Not a valid virtualenv."
        exit 1
    fi
    echo "Using existing virtualenv: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
else
    VENV_DIR=".venv"
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        echo "Creating virtualenv at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    echo "Installing requirements..."
    pip install -q -r requirements.txt
fi

# Force CPU — MLP policies are faster on CPU than GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"

export NETWORKS_DIR="${NETWORKS_DIR:-${DATA_DIR}/networks}"
export OUTPUT_BASE="${OUTPUT_BASE:-${DATA_DIR}/networks}"
export RESULTS_DIR="${RESULTS_DIR:-${DATA_DIR}/results}"

MODE_LABEL="full"
if [[ "$TEST_MODE" == "true" ]]; then
    MODE_LABEL="test"
fi

echo "============================================================"
echo "  Article 1 Reproduction Pipeline  [${MODE_LABEL}]"
echo "  Python:   $(python --version 2>&1) ($(which python))"
echo "  Batch:    ${BATCH:-article1-all-conditions}"
echo "  Seeds:    ${SEEDS:-42,123,456,789,1024,2048,4096,8192,16384,32768}"
echo "  Episodes: ${EPISODES:-100}"
echo "  Networks: ${NETWORKS_DIR}"
echo "  Results:  ${RESULTS_DIR}"
echo "============================================================"
echo ""

if [[ "$SKIP_TRAINING" == "false" ]]; then
    scripts/01-train.sh
fi

scripts/02-eval-indist.sh
scripts/03-eval-corruption.sh
scripts/04-eval-ood.sh
scripts/05-analyze.sh
scripts/06-summary.sh
