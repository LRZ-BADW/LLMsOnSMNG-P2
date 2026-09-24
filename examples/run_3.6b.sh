# 3.6B GPT preset (single node / 8 tiles by default).
# Run from the Megatron-DeepSpeed directory after sourcing env/set_env.sh and
# generating hostfiles. Override any variable on the command line, e.g.:
#   TRAIN_ITER=20 bash ../examples/run_3.6b.sh
echo "Generate hostfiles (source examples/generate_hostfile.sh) before training!"
export MODEL="3.6b"
export WORLD_SIZE=${WORLD_SIZE:-8}
export MICRO_BATCH=${MICRO_BATCH:-8}
export NLAYERS=${NLAYERS:-30}
export HIDDEN=${HIDDEN:-3072}
export HEADS=${HEADS:-32}
export SEQ=${SEQ:-2048}
export TRAIN_ITER=${TRAIN_ITER:-10}
export ZERO_STAGE=${ZERO_STAGE:-2}
export DTYPE=${DTYPE:-bf16}
export TP=${TP:-1}
export PP=${PP:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))

if [ -n "$RUNS_DIR" ]; then
    mkdir -p "$RUNS_DIR"
    params_csv="${RUNS_DIR}/${MODEL}-${SLURM_JOB_ID}-params.csv"
    [ -f "$params_csv" ] || echo "JOB_ID,PP,TP,WORLD_SIZE,MICRO_BATCH,GRAD_ACC_STEPS,GLOBAL_BATCH,ZERO_STAGE,TRAIN_ITER" > "$params_csv"
    echo "${SLURM_JOB_ID},${PP},${TP},${WORLD_SIZE},${MICRO_BATCH},${GRAD_ACC_STEPS},${GLOBAL_BATCH},${ZERO_STAGE},${TRAIN_ITER}" >> "$params_csv"
fi

bash $LLM_DK_DIR/examples/gpt.sh --no-query-key-layer-scaling $@
