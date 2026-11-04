#!/usr/bin/env bash
# Compatibility entry point for the configuration-driven benchmark runner.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:---help}
if [[ $# -gt 0 ]]; then
    shift
fi
case "$MODE" in
    ucm-seed) MODE=posix-seed ;;
    ucm-run) MODE=posix-run ;;
esac
OPTIONS=()
if [[ ${DRY_RUN:-0} == 1 ]]; then
    OPTIONS+=(--dry-run)
fi
export VLLMBENCH_LEGACY_ENV=1
exec "${PYTHON:-python}" "$SCRIPT_DIR/bench.py" "$MODE" "${BENCH_CONFIG:-$SCRIPT_DIR/bench.json}" "${OPTIONS[@]}" "$@"
