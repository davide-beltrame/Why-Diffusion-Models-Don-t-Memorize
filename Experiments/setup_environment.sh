#!/usr/bin/env bash
# Backward-compatible alias for the non-interactive setup entry point.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
    echo "Select installation type: 1) GPU (CUDA 12.1)  2) CPU"
    read -r -p "Enter choice (1 or 2): " choice
    case "$choice" in
        1) set -- --gpu ;;
        2) set -- --cpu ;;
        *) echo "Invalid choice." >&2; exit 1 ;;
    esac
fi
exec bash "$SCRIPT_DIR/setup_env.sh" "$@"
