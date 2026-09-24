#!/bin/bash
# Build and launch the Megatron-DeepSpeed GPT pre-training command.
# Not called directly — invoked by the run_*.sh presets, which export the
# model/parallelism envs first. Must be run from the Megatron-DeepSpeed dir
# (it calls pretrain_gpt.py) with LLM_DK_DIR pointing at this repo root.

VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt
DATA_PATH=dataset/BookCorpusDataset_text_document
DTYPE=${DTYPE:-bf16}

# Hostfile paths (produced by generate_hostfile.sh, keyed by SLURM_JOB_ID).
hostfile_suffix=${SLURM_JOB_ID:+_${SLURM_JOB_ID}}
hostfile_deepspeed=$LLM_DK_DIR/examples/hostfiles/hostfile_deepspeed${hostfile_suffix}
hostfile_mpich=$LLM_DK_DIR/examples/hostfiles/hostfile_mpich${hostfile_suffix}

# Pin the rendezvous explicitly rather than relying on DeepSpeed's default
# election, which can race when multiple jobs launch back-to-back.
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
export MASTER_PORT=${MASTER_PORT:-29500}

# Disabling tensor/pipeline parallelism by default.
TP=${TP:-1}
PP=${PP:-1}

# Model: default 3.6b
NLAYERS=${NLAYERS:-30}
HIDDEN=${HIDDEN:-3072}
HEADS=${HEADS:-32}
SEQ=${SEQ:-2048}
TRAIN_ITER=${TRAIN_ITER:-10}

WORLD_SIZE=${WORLD_SIZE:-8}
MICRO_BATCH=${MICRO_BATCH:-8}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}

ZERO_STAGE=${ZERO_STAGE:-2}

DS_CONFIG=$LLM_DK_DIR/examples/"ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_gas${GRAD_ACC_STEPS}_pp${PP}_tp${TP}_ws${WORLD_SIZE}_${DTYPE}.json"
bash $LLM_DK_DIR/examples/generate_config.sh ${DS_CONFIG} || exit 1

OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}_`date +%m%d%H%M%S`_${HOSTNAME}
mkdir -p $OUTPUT_DIR
echo "!!!Please see logs at ${OUTPUT_DIR}"

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ $PP == 1 ]; then
   ds_args=" --no-pipeline-parallel ${ds_args}"
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
# Activation checkpointing is provided by Megatron (see --checkpoint-activations below).
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

# Take custom args.
custom_args=" $@"

# Launcher setting.
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher="--master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    launcher="--master_addr $MASTER_ADDR --master_port $MASTER_PORT --force_multi --hostfile $hostfile_deepspeed --launcher=${LAUNCHER} --launcher_args='-hostfile ${hostfile_mpich}'"
fi

CCL=${CCL:-ccl}

run_cmd="
    deepspeed $launcher pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITER \
    --lr 0.00015 \
    --lr-warmup-fraction .01 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 100 \
    --eval-interval 100 \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --save-interval 500 \
    --split 100,0,0 \
    --$DTYPE \
    --checkpoint-activations \
    --deepspeed-activation-checkpointing
    $ds_args \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --distributed-backend $CCL \
    --num-workers 0 \
    $custom_args \
    |& tee $OUTPUT_DIR/output.log
    "

echo ${run_cmd}
eval ${run_cmd}
set +x
