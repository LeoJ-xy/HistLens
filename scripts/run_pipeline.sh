#!/bin/bash
# ============================================================================
# HistSAE pipeline wrapper
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${PROJECT_ROOT}/configs/exp/default.yaml"
STAGES=""
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 [--config <path>] [--stages <a,b,c>] [--resume|--no-resume] [--force] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --stages)
            STAGES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CMD=("python" "${PROJECT_ROOT}/src/pipeline/run.py" "--config" "${CONFIG_FILE}")
if [[ -n "${STAGES}" ]]; then
    CMD+=("--stages" "${STAGES}")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
