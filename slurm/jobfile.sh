#!/bin/bash
# Generic SBATCH jobfile for the batch submission layer. Carries no
# --account / -p / -N / -t / -J / -o of its own — submit.sh (or sweep.sh /
# scaling.sh, which call submit.sh) supplies all of those as sbatch flags, so
# this one file serves every model, queue and site.
#
# Not meant to be sbatch'd directly. Use slurm/submit.sh, which also exports
# the variables this script expects to find in its environment:
#   MODEL WORLD_SIZE TP PP MICRO_BATCH GRAD_ACC_STEPS ZERO_STAGE TRAIN_ITER
#   DTYPE (optional)

set -eu

# ---- site configuration -----------------------------------------------------
site_conf="$(dirname "${BASH_SOURCE[0]}")/site.conf"
if [ ! -f "$site_conf" ]; then
    echo "Missing $site_conf — copy slurm/site.conf.example and fill it in." >&2
    exit 1
fi
. "$site_conf"
# -----------------------------------------------------------------------------

echo "Running ${MODEL} on ${SLURM_NNODES} node(s), job ${SLURM_JOB_ID}..."

source "${CONDA_ROOT}"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
source "${REPO_DIR}/env/set_env.sh"

cd "${REPO_DIR}/examples"
source "${REPO_DIR}/examples/generate_hostfile.sh"

if [ -n "${CHECKPOINT_ROOT_DIR:-}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_ROOT_DIR}/${SLURM_JOB_ID}"
    mkdir -p "${CHECKPOINT_DIR}"
    export CHECKPOINT_DIR
fi

cd "${REPO_DIR}/Megatron-DeepSpeed"

echo "=========================================="
echo "MODEL=${MODEL}"
echo "PP=${PP}  TP=${TP}  WORLD_SIZE=${WORLD_SIZE}"
echo "MICRO_BATCH=${MICRO_BATCH}  GRAD_ACC_STEPS=${GRAD_ACC_STEPS}  ZERO_STAGE=${ZERO_STAGE}"
echo "TRAIN_ITER=${TRAIN_ITER}  DTYPE=${DTYPE:-bf16}"
[ -n "${CHECKPOINT_DIR:-}" ] && echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "=========================================="

exec bash "${REPO_DIR}/examples/run_${MODEL}.sh"
