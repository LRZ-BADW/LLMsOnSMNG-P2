#!/bin/bash
# Runtime environment for Intel XPU GPT pre-training (Megatron-DeepSpeed + oneCCL).
# Source this on the allocated node(s) after `conda activate <env>`:
#   source env/set_env.sh          # 8 tiles (one PVC OAM node)
#   source env/set_env.sh <N>      # N local device (tile) indices
#
# Sets a linear ZE_AFFINITY_MASK over the local tiles plus the oneCCL / Level-Zero
# variables that matter on PVC. See docs/TUNING.md for what each one does.

export LLM_DK_DIR=$(pwd)

################# ZE_AFFINITY_MASK (flat tile indices) #################
# Local flat indices 0..(n-1); default 8 tiles per node.
num_local_devices=8

generate_mask () {
    local total=$1
    local out=""
    for ((i=0;i<total;i++)); do
        out+=$i
        (( i < total-1 )) && out+=","
    done
    echo "$out"
}

# Allow override via first argument.
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    num_local_devices=$1
fi
ZE_AFFINITY_MASK=$(generate_mask $num_local_devices)
export ZE_AFFINITY_MASK
echo "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"

################# ulimit #################
set_max_ulimit () {
    local max_ulimit
    max_ulimit=$(ulimit -H -n)
    [ "$max_ulimit" = "unlimited" ] && max_ulimit=$(ulimit -S -n)
    ulimit -n "$max_ulimit"
    echo "Open file limit set to: $max_ulimit"
}
set_max_ulimit

################# MPI module (site-specific) #################
# Load your site's MPI module here. The module names below are placeholders —
# map them to whatever provides Intel MPI / MPICH on your cluster.
# module load <your-mpi-module>
# module load gcc

################# Core env vars #################
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1

# Uncomment if your launcher needs explicit oneCCL transport settings:
# export CCL_ATL_TRANSPORT=mpi
# export CCL_PROCESS_LAUNCHER=mpi

export CCL_OP_SYNC=1
export CCL_SKIP_SCHEDULER=1
export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
# export I_MPI_DEBUG=7   # raise for debugging; keep low/unset in production

echo "Environment ready."
