#!/bin/bash
# Submit one batch training job: validates the config, assembles sbatch flags
# and an --export list, and calls `sbatch slurm/jobfile.sh`. Called directly,
# or by sweep.sh / scaling.sh for one row at a time.
#
# Usage:
#   slurm/submit.sh --model 20b --world-size 32 --tp 8 --pp 4 \
#     --micro-batch 1 --grad-acc 8 --zero 1 \
#     [--time HH:MM:SS] [--train-iter N] [--dtype bf16|fp16] [--dry-run]

set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
site_conf="${script_dir}/site.conf"
if [ ! -f "$site_conf" ]; then
    echo "Missing $site_conf — copy slurm/site.conf.example and fill it in." >&2
    exit 1
fi
. "$site_conf"

# NLAYERS per model preset, mirroring the defaults in examples/run_*.sh.
# Used only to validate --pp before submitting. A case statement rather than
# an associative array: macOS ships bash 3.2, which lacks `declare -A`.
nlayers_for_model() {
    case "$1" in
        3.6b) echo 30 ;;
        20b) echo 44 ;;
        175b) echo 96 ;;
        *) echo "" ;;
    esac
}

TRAIN_ITER=10
DTYPE=bf16
TIME=""
DRY_RUN=0

usage() {
    cat >&2 <<EOF
Usage: $0 --model M --world-size N --tp T --pp P --micro-batch B --grad-acc G --zero Z
          [--time HH:MM:SS] [--train-iter N] [--dtype bf16|fp16] [--dry-run]
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --world-size) WORLD_SIZE="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --micro-batch) MICRO_BATCH="$2"; shift 2 ;;
        --grad-acc) GRAD_ACC_STEPS="$2"; shift 2 ;;
        --zero) ZERO_STAGE="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --train-iter) TRAIN_ITER="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

for var in MODEL WORLD_SIZE TP PP MICRO_BATCH GRAD_ACC_STEPS ZERO_STAGE; do
    if [ -z "${!var:-}" ]; then
        echo "Missing required flag for $var" >&2
        usage
    fi
done

nlayers="$(nlayers_for_model "$MODEL")"
if [ -z "$nlayers" ]; then
    echo "Unknown --model '$MODEL' (known: 3.6b, 20b, 175b)" >&2
    exit 1
fi
if [ "$TP" -gt "$TILES_PER_NODE" ]; then
    echo "--tp=$TP exceeds TILES_PER_NODE=$TILES_PER_NODE (TP must fit on one node)" >&2
    exit 1
fi
if [ $(( nlayers % PP )) -ne 0 ]; then
    echo "--pp=$PP does not divide NLAYERS=$nlayers for model '$MODEL'" >&2
    exit 1
fi
if [ $(( WORLD_SIZE % TILES_PER_NODE )) -ne 0 ]; then
    echo "--world-size=$WORLD_SIZE is not a multiple of TILES_PER_NODE=$TILES_PER_NODE" >&2
    exit 1
fi

NUM_NODES=$(( WORLD_SIZE / TILES_PER_NODE ))

partition="$PARTITION"
if [ -n "${LARGE_NODE_THRESHOLD:-}" ] && [ "$NUM_NODES" -ge "$LARGE_NODE_THRESHOLD" ]; then
    partition="$PARTITION_LARGE"
fi

time="${TIME:-$DEFAULT_TIME}"
job_name="${MODEL}-tp${TP}-pp${PP}-ws${WORLD_SIZE}"

mkdir -p "$RUNS_DIR"
out_log="${RUNS_DIR}/${job_name}-%j.out"

export_list="MODEL=${MODEL},WORLD_SIZE=${WORLD_SIZE},TP=${TP},PP=${PP},MICRO_BATCH=${MICRO_BATCH},GRAD_ACC_STEPS=${GRAD_ACC_STEPS},ZERO_STAGE=${ZERO_STAGE},TRAIN_ITER=${TRAIN_ITER},DTYPE=${DTYPE},RUNS_DIR=${RUNS_DIR}"

sbatch_cmd=(sbatch
    "--account=${SLURM_ACCOUNT}"
    "--partition=${partition}"
    "--nodes=${NUM_NODES}"
    "--time=${time}"
    "--job-name=${job_name}"
    "--output=${out_log}"
    "--export=ALL,${export_list}"
)
if [ -n "${EXTRA_SBATCH_ARGS:-}" ]; then
    # Intentionally unquoted: EXTRA_SBATCH_ARGS is a space-separated list of flags.
    sbatch_cmd+=( ${EXTRA_SBATCH_ARGS} )
fi
sbatch_cmd+=( "${script_dir}/jobfile.sh" )

echo "${sbatch_cmd[@]}"
if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

"${sbatch_cmd[@]}"
