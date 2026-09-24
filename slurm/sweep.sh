#!/bin/bash
# Submit one job per row of a CSV sweep config, via submit.sh.
#
# Usage: slurm/sweep.sh <csv> [--dry-run] [--time HH:MM:SS] [--train-iter N] [--dtype bf16|fp16]
#
# CSV header: MODEL,PP,TP,MICRO_BATCH,GRAD_ACC_STEPS,ZERO_STAGE,WORLD_SIZE
# See slurm/configs/example_*.csv.

set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <csv> [--dry-run] [--time HH:MM:SS] [--train-iter N] [--dtype bf16|fp16]" >&2
    exit 1
}

[ $# -ge 1 ] || usage
csv="$1"; shift
[ -f "$csv" ] || { echo "No such file: $csv" >&2; exit 1; }

extra_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) extra_args+=( --dry-run ); shift ;;
        --time) extra_args+=( --time "$2" ); shift 2 ;;
        --train-iter) extra_args+=( --train-iter "$2" ); shift 2 ;;
        --dtype) extra_args+=( --dtype "$2" ); shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

expected_header="MODEL,PP,TP,MICRO_BATCH,GRAD_ACC_STEPS,ZERO_STAGE,WORLD_SIZE"
header="$(head -n1 "$csv" | tr -d '\r')"
if [ "$header" != "$expected_header" ]; then
    echo "Bad CSV header in $csv" >&2
    echo "  expected: $expected_header" >&2
    echo "  got:      $header" >&2
    exit 1
fi

row=0
while IFS=, read -r model pp tp micro_batch grad_acc zero world_size; do
    [ -z "${model}${pp}${tp}${micro_batch}${grad_acc}${zero}${world_size}" ] && continue
    row=$((row + 1))
    echo "--- row ${row}: model=${model} pp=${pp} tp=${tp} micro_batch=${micro_batch} grad_acc=${grad_acc} zero=${zero} world_size=${world_size} ---"
    "${script_dir}/submit.sh" \
        --model "$model" --world-size "$world_size" --tp "$tp" --pp "$pp" \
        --micro-batch "$micro_batch" --grad-acc "$grad_acc" --zero "$zero" \
        "${extra_args[@]}"
done < <(tail -n +2 "$csv" | tr -d '\r')

echo "Submitted ${row} job(s) from ${csv}."
