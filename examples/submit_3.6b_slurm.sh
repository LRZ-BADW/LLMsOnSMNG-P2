#!/bin/bash
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH -p <PARTITION>
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J 3.6b
#SBATCH -o %x-%j.out
# Most sites also need an explicit GPU/task request, e.g.:
##SBATCH --gres=gpu:4
##SBATCH --ntasks-per-node=8
#
# Batch variant of the 3.6B warm-up. Edit the SBATCH directives above for your
# site (account, partition, optional --qos / --reservation), set REPO_DIR /
# CONDA_ENV below, then: sbatch examples/submit_3.6b_slurm.sh
#
# Override model/parallelism knobs by exporting them before the run_3.6b.sh call,
# e.g. WORLD_SIZE, PP, TP, MICRO_BATCH, GRAD_ACC_STEPS, ZERO_STAGE, TRAIN_ITER.

# ---- site configuration -----------------------------------------------------
REPO_DIR=${REPO_DIR:-$HOME/llm-devkit-pvc}   # this repo's root
CONDA_ENV=${CONDA_ENV:-gpt_pt28}
# -----------------------------------------------------------------------------

echo "Running 3.6B on ${SLURM_NNODES} node(s)..."

export WORLD_SIZE=${WORLD_SIZE:-$(( SLURM_NNODES * 8 ))}

source "$HOME/miniforge3/bin/activate"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
source "${REPO_DIR}/env/set_env.sh"

# Regenerate hostfiles for this allocation.
rm -f "${REPO_DIR}/examples/hostfile_deepspeed" "${REPO_DIR}/examples/hostfile_mpich"
cd "${REPO_DIR}/examples"
source "${REPO_DIR}/examples/generate_hostfile.sh"

cd "${REPO_DIR}/Megatron-DeepSpeed"

echo "=========================================="
echo "PP=${PP}  TP=${TP}  WORLD_SIZE=${WORLD_SIZE}"
echo "MICRO_BATCH=${MICRO_BATCH}  GRAD_ACC_STEPS=${GRAD_ACC_STEPS}  ZERO_STAGE=${ZERO_STAGE}"
echo "=========================================="

bash "${REPO_DIR}/examples/run_3.6b.sh"
