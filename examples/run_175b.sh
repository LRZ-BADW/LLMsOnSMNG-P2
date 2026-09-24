# 175B GPT preset — validated 3D-parallel config for PVC OAM nodes (8 tiles/node).
# 175B needs a large model-parallel mesh: one replica spans TP=8 x PP=16 = 128 tiles
# = 16 nodes. Defaults (WORLD_SIZE=128) run a single replica; scale out by adding
# data-parallel replicas (multiply WORLD_SIZE by the replica count).
# TP=8 stays within one node (high-bandwidth Xe Link intra-node fabric); PP=16 divides
# NLAYERS=96 (96/16=6 layers/stage); MICRO_BATCH=3 and GRAD_ACC_STEPS=100 keep the
# pipeline full. ZeRO-1 (optimizer-state sharding) is the right stage for scale-out;
# ZeRO-2/3 add gradient/parameter collectives whose cost dominates over the fabric.
# Run from the Megatron-DeepSpeed directory after sourcing env/set_env.sh and
# generating hostfiles for all nodes. Examples:
#   bash ../examples/run_175b.sh                                   # 1 replica, 16 nodes
#   WORLD_SIZE=256 bash ../examples/run_175b.sh                    # 2 replicas (DP=2), 32 nodes
export MODEL="175b"
export WORLD_SIZE=${WORLD_SIZE:-128}
export MICRO_BATCH=${MICRO_BATCH:-3}
export NLAYERS=${NLAYERS:-96}
export HIDDEN=${HIDDEN:-12288}
export HEADS=${HEADS:-96}
export SEQ=${SEQ:-2048}
export TRAIN_ITER=${TRAIN_ITER:-10}
export ZERO_STAGE=${ZERO_STAGE:-1}
export DTYPE=${DTYPE:-bf16}
export TP=${TP:-8}
export PP=${PP:-16}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-100}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))

if [ -n "$RUNS_DIR" ]; then
    mkdir -p "$RUNS_DIR"
    params_csv="${RUNS_DIR}/${MODEL}-${SLURM_JOB_ID}-params.csv"
    [ -f "$params_csv" ] || echo "JOB_ID,PP,TP,WORLD_SIZE,MICRO_BATCH,GRAD_ACC_STEPS,GLOBAL_BATCH,ZERO_STAGE,TRAIN_ITER" > "$params_csv"
    echo "${SLURM_JOB_ID},${PP},${TP},${WORLD_SIZE},${MICRO_BATCH},${GRAD_ACC_STEPS},${GLOBAL_BATCH},${ZERO_STAGE},${TRAIN_ITER}" >> "$params_csv"
fi

echo "Generate hostfiles for $((${WORLD_SIZE}/8)) nodes before training!"
bash $LLM_DK_DIR/examples/gpt.sh $@
