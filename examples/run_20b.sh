# 20B GPT preset — validated 3D-parallel config for PVC OAM nodes (8 tiles/node).
# 20B exceeds a single tile's 64 GB HBM, so it needs model parallelism. Defaults
# target 4 nodes (WORLD_SIZE=32): TP=8 (kept within one node), PP=4, and ZeRO-1
# sharded data parallelism.
# ZeRO-1 (shard optimizer states only) is preferred over ZeRO-2/3 for multi-node
# runs: higher stages add gradient/parameter all-gathers that dominate over the
# interconnect. Run from the Megatron-DeepSpeed directory after sourcing
# env/set_env.sh and generating hostfiles. Scale out with data-parallel replicas,
# e.g. 8 nodes (DP=2): WORLD_SIZE=64 bash ../examples/run_20b.sh
export MODEL="20b"
export WORLD_SIZE=${WORLD_SIZE:-32}
export MICRO_BATCH=${MICRO_BATCH:-1}
export NLAYERS=${NLAYERS:-44}
export HIDDEN=${HIDDEN:-6144}
export HEADS=${HEADS:-64}
export SEQ=${SEQ:-2048}
export TRAIN_ITER=${TRAIN_ITER:-10}
export ZERO_STAGE=${ZERO_STAGE:-1}
export DTYPE=${DTYPE:-bf16}
export TP=${TP:-8}
export PP=${PP:-4}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))

if [ -n "$RUNS_DIR" ]; then
    mkdir -p "$RUNS_DIR"
    params_csv="${RUNS_DIR}/${MODEL}-${SLURM_JOB_ID}-params.csv"
    [ -f "$params_csv" ] || echo "JOB_ID,PP,TP,WORLD_SIZE,MICRO_BATCH,GRAD_ACC_STEPS,GLOBAL_BATCH,ZERO_STAGE,TRAIN_ITER" > "$params_csv"
    echo "${SLURM_JOB_ID},${PP},${TP},${WORLD_SIZE},${MICRO_BATCH},${GRAD_ACC_STEPS},${GLOBAL_BATCH},${ZERO_STAGE},${TRAIN_ITER}" >> "$params_csv"
fi

echo "Generate hostfiles for $((${WORLD_SIZE}/8)) nodes before training!"
bash $LLM_DK_DIR/examples/gpt.sh $@
