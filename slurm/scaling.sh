#!/bin/bash
# 175B weak/strong scaling sweep. Submits one job per factor via submit.sh.
#
# Usage: slurm/scaling.sh --mode weak|strong --factors "1 2 4 8"
#          [--world-size N] [--tp T] [--pp P] [--micro-batch B] [--zero Z]
#          [--time HH:MM:SS] [--train-iter N] [--dtype bf16|fp16] [--dry-run]
#
# Base config is the examples/run_175b.sh defaults: WORLD_SIZE=128 TP=8 PP=16
# MICRO_BATCH=3. Each factor f scales WORLD_SIZE to BASE_WORLD_SIZE * f.
#
# --mode weak holds GRAD_ACC_STEPS fixed at 100 (work per accelerator constant;
# global batch grows with node count).
# --mode strong holds the global batch fixed at 2520 and derives GRAD_ACC_STEPS
# backward from it (fixes a real defect in the private 1tStrongScaling.sh,
# which was a copy of the weak-scaling script and never recomputed
# GRAD_ACC_STEPS, so it silently ran weak scaling under a "strong" label).
#
# No --grad-acc flag: it is always derived, that derivation is the point.

set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="175b"
BASE_WORLD_SIZE=128
TP=8
PP=16
MICRO_BATCH=3
ZERO_STAGE=1
WEAK_GRAD_ACC_STEPS=100
STRONG_GLOBAL_BATCH=2520

TIME=""
TRAIN_ITER=""
DTYPE=""
DRY_RUN=0
MODE=""
FACTORS=""

usage() {
    cat >&2 <<EOF
Usage: $0 --mode weak|strong --factors "1 2 4 8"
          [--world-size N] [--tp T] [--pp P] [--micro-batch B] [--zero Z]
          [--time HH:MM:SS] [--train-iter N] [--dtype bf16|fp16] [--dry-run]
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --factors) FACTORS="$2"; shift 2 ;;
        --world-size) BASE_WORLD_SIZE="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --micro-batch) MICRO_BATCH="$2"; shift 2 ;;
        --zero) ZERO_STAGE="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --train-iter) TRAIN_ITER="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

case "$MODE" in
    weak|strong) ;;
    *) echo "--mode must be 'weak' or 'strong', got '${MODE:-}'" >&2; usage ;;
esac
[ -n "$FACTORS" ] || usage

extra_args=()
[ -n "$TIME" ] && extra_args+=( --time "$TIME" )
[ -n "$TRAIN_ITER" ] && extra_args+=( --train-iter "$TRAIN_ITER" )
[ -n "$DTYPE" ] && extra_args+=( --dtype "$DTYPE" )
[ "$DRY_RUN" -eq 1 ] && extra_args+=( --dry-run )

for f in $FACTORS; do
    new_world_size=$(( BASE_WORLD_SIZE * f ))

    if [ "$MODE" = "weak" ]; then
        grad_acc="$WEAK_GRAD_ACC_STEPS"
    else
        numerator=$(( STRONG_GLOBAL_BATCH * TP * PP ))
        denominator=$(( new_world_size * MICRO_BATCH ))
        if [ $(( numerator % denominator )) -ne 0 ]; then
            echo "strong mode: GRAD_ACC_STEPS = ${numerator} / ${denominator} is not a whole number for factor ${f}" >&2
            exit 1
        fi
        grad_acc=$(( numerator / denominator ))
    fi

    echo "--- factor ${f}: world-size=${new_world_size} grad-acc=${grad_acc} (mode=${MODE}) ---"
    "${script_dir}/submit.sh" \
        --model "$MODEL" --world-size "$new_world_size" --tp "$TP" --pp "$PP" \
        --micro-batch "$MICRO_BATCH" --grad-acc "$grad_acc" --zero "$ZERO_STAGE" \
        "${extra_args[@]}"
done
