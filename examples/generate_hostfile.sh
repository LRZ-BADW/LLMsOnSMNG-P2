#!/bin/bash
# Generate deepspeed + mpich hostfiles from the current SLURM allocation.
# Source (don't just run) from the examples/ directory, on an allocated node:
#   cd examples && source generate_hostfile.sh
# Falls back to the local hostname when not inside a SLURM job.

# Directory to store the host files, keyed by SLURM_JOB_ID so concurrent
# jobs don't clobber each other's hostfiles.
examples_dir=${LLM_DK_DIR:+$LLM_DK_DIR/examples}
examples_dir=${examples_dir:-$(pwd)}
hostfile_dir="$examples_dir/hostfiles"
mkdir -p "$hostfile_dir"

if [ -n "$SLURM_JOB_ID" ]; then
    suffix="_${SLURM_JOB_ID}"
else
    suffix=""
fi
hostfile_mpich="$hostfile_dir/hostfile_mpich${suffix}"
hostfile_deepspeed="$hostfile_dir/hostfile_deepspeed${suffix}"

# Use the SLURM nodelist if present, else the current hostname.
if [ -n "$SLURM_NODELIST" ]; then
    nodes=$(scontrol show hostnames $SLURM_NODELIST)
else
    nodes=$(hostname)
fi

# Truncate existing hostfiles.
if [ -f "$hostfile_mpich" ]; then
    cat /dev/null > "$hostfile_mpich"
fi
if [ -f "$hostfile_deepspeed" ]; then
    cat /dev/null > "$hostfile_deepspeed"
fi

# One line per node: deepspeed wants "host slots=N", mpich wants just the host.
echo "$nodes" | while read -r node; do
    echo "$node slots=8" >> "$hostfile_deepspeed"
    echo "$node"         >> "$hostfile_mpich"
done

echo "Hostfiles generated successfully!"
