#!/usr/bin/env bash
# A3 DP2 comparison using vllm bench serve and its random dataset.
# Usage: bash run_a3_pc_bench.sh {ucm-seed|ucm-run|hbm}
# Each invocation covers both 32k/40 and 128k/10.
# Restart once between ucm-seed and ucm-run, preserving the Posix files.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_a3_pc_bench.sh ACTION
  ucm-seed   Seed both cases on DP0 and DP1.
  ucm-run    Measure both cases after restarting the UCM service.
  hbm        Warm both DP ranks and measure, automatically for each case.

Environment overrides
  MODEL=/data/models/GLM-5.3-Flash-w8a8
  SERVED_MODEL=glm HOST=localhost PORT=18081
  SEED=2004          Base seed; 128k uses SEED + 1.
  BLOCK_TOKENS=1      Set to the service's actual LCM block size to align seeds.
  RESULT_DIR=./a3_pc_results
  DRY_RUN=1          Print commands without sending requests.

UCM workflow
  ucm-seed -> wait for UCM dump completion -> stop service -> restart
  with the same Posix directory and key configuration -> ucm-run

HBM-PC workflow
  hbm, on the HBM-PC baseline service

All phases use the same seed and number of prompts within a case.
Seeding and warming explicitly cover both DP ranks through request headers.
Measured runs send one batch of 40 or 10 requests, with automatic DP routing.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# != 1 ]]; then
    usage >&2
    exit 2
fi

ACTION=$1
case "$ACTION" in
    ucm-seed|ucm-run|hbm) ;;
    *) usage >&2; exit 2 ;;
esac

MODEL=${MODEL:-/data/models/GLM-5.3-Flash-w8a8}
SERVED_MODEL=${SERVED_MODEL:-glm}
HOST=${HOST:-localhost}
PORT=${PORT:-18081}
BASE_SEED=${SEED:-2004}
BLOCK_TOKENS=${BLOCK_TOKENS:-1}
RESULT_DIR=${RESULT_DIR:-./a3_pc_results}
DRY_RUN=${DRY_RUN:-0}
BENCH_OPTIONS=()

if [[ ! $BLOCK_TOKENS =~ ^[1-9][0-9]*$ ]]; then
    echo 'BLOCK_TOKENS must be a positive integer.' >&2
    exit 2
fi
if ((BLOCK_TOKENS > 32768 * 9 / 10)); then
    echo 'BLOCK_TOKENS is too large for the 32k case.' >&2
    exit 2
fi

if [[ $DRY_RUN != 1 ]]; then
    # Some versions show only argument groups with plain --help.
    BENCH_HELP=$(vllm bench serve --help=all 2>&1) || BENCH_HELP=$(vllm bench serve --help)
    for option in --header --random-prefix-len; do
        if [[ $BENCH_HELP != *"$option"* ]]; then
            echo "Installed vllm bench serve does not support $option." >&2
            exit 2
        fi
    done
    if [[ $BENCH_HELP == *"--ready-check-timeout-sec"* ]]; then
        BENCH_OPTIONS+=(--ready-check-timeout-sec 0)
    else
        echo 'This bench version may run an initial prompt probe and warm its cache before measurement.' >&2
    fi
    if [[ $BENCH_HELP == *"--num-warmups"* ]]; then
        BENCH_OPTIONS+=(--num-warmups 0)
    fi
    mkdir -p "$RESULT_DIR"
fi
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$

run_bench() {
    local label=$1 input=$2 prefix=$3 output=$4 concurrency=$5 rank=${6:-}
    local result_name="${CASE}_${label}_${RUN_ID}"
    local cmd=(
        vllm bench serve
        --backend vllm
        --model "$MODEL"
        --served-model-name "$SERVED_MODEL"
        --host "$HOST" --port "$PORT"
        --seed "$SEED"
        --dataset-name random
        --num-prompts "$NUM_PROMPTS"
        --random-input-len "$input"
        --random-prefix-len "$prefix"
        --random-range-ratio 0
        --random-output-len "$output"
        --max-concurrency "$concurrency"
        --request-rate inf
        --ignore-eos
        "${BENCH_OPTIONS[@]}"
        --save-result --save-detailed
        --result-dir "$RESULT_DIR"
        --result-filename "${result_name}.json"
    )
    if [[ -n $rank ]]; then
        cmd+=(--header "X-data-parallel-rank=$rank")
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ $DRY_RUN == 1 ]]; then
        return
    fi
    "${cmd[@]}" 2>&1 | tee "$RESULT_DIR/${result_name}.log"
    # bench can exit successfully even when individual requests fail.
    python - "$RESULT_DIR/${result_name}.json" "$NUM_PROMPTS" "$output" <<'PY'
import json
import sys

path, expected_count, expected_output = sys.argv[1:]
expected_count, expected_output = int(expected_count), int(expected_output)
with open(path, encoding="utf-8") as source:
    result = json.load(source)
if result.get("completed") != expected_count or result.get("failed", 0) != 0:
    raise SystemExit(f"Incomplete benchmark, inspect {path}")
lengths = result.get("output_lens", [])
if len(lengths) != expected_count or any(n != expected_output for n in lengths):
    raise SystemExit(f"Unexpected output lengths, inspect {path}")
print(f"Validated {expected_count} completed requests, {expected_output} output tokens each.")
PY
}

if [[ $ACTION == ucm-run ]]; then
    echo 'Both cases require a restarted UCM service with the seeded Posix files.'
fi
for CASE in 32k 128k; do
    case "$CASE" in
        32k) INPUT_LEN=32768; NUM_PROMPTS=40; SEED=$BASE_SEED ;;
        128k) INPUT_LEN=131072; NUM_PROMPTS=10; SEED=$((BASE_SEED + 1)) ;;
    esac
    HIT_LEN=$((INPUT_LEN * 9 / 10 / BLOCK_TOKENS * BLOCK_TOKENS))
    SUFFIX_LEN=$((INPUT_LEN - HIT_LEN))
    PER_DP_CONCURRENCY=$((NUM_PROMPTS / 2))
    printf 'action=%s case=%s input=%s prompts=%s seed=%s seed_tokens=%s block_tokens=%s\n' \
        "$ACTION" "$CASE" "$INPUT_LEN" "$NUM_PROMPTS" "$SEED" "$HIT_LEN" "$BLOCK_TOKENS"

    case "$ACTION" in
        ucm-seed)
            for rank in 0 1; do
                run_bench "ucm_seed_dp${rank}" "$HIT_LEN" 0 1 "$PER_DP_CONCURRENCY" "$rank"
            done
            ;;
        ucm-run)
            run_bench ucm_posix "$INPUT_LEN" 0 512 "$NUM_PROMPTS"
            ;;
        hbm)
            # Keep NUM_PROMPTS unchanged because random prefix generation follows
            # per-request sampling. A tiny suffix leaves only the shared prefix hot.
            for rank in 0 1; do
                run_bench "hbm_warm_dp${rank}" 1 "$HIT_LEN" 1 "$PER_DP_CONCURRENCY" "$rank"
            done
            run_bench hbm_pc "$SUFFIX_LEN" "$HIT_LEN" 512 "$NUM_PROMPTS"
            ;;
    esac
done
if [[ $ACTION == ucm-seed ]]; then
    echo 'Both cases are seeded. Wait for UCM dump completion, then stop and restart with the same Posix files.'
fi
