#!/usr/bin/env bash
set -euo pipefail

MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) MODE="gpu"; shift ;;
        --cpu) MODE="cpu"; shift ;;
        *) echo "Usage: bash setup_env.sh --gpu | --cpu"; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: bash setup_env.sh --gpu | --cpu"
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not installed or not in PATH." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-memorization}"
if [[ "$MODE" == "gpu" ]]; then
    ENV_FILE="$SCRIPT_DIR/environment.yml"
else
    ENV_FILE="$SCRIPT_DIR/environment_cpu.yml"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda run -n "$ENV_NAME" python -c \
    'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
echo "Environment ready. Activate with: conda activate $ENV_NAME"
