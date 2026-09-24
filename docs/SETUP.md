# Setup — full recipe

End-to-end recipe for GPT pre-training with Megatron-DeepSpeed on a SLURM cluster of
Intel Data Center GPU Max 1550 (PVC) OAM nodes — 4 dual-tile cards = 8 tiles per node,
64 GB HBM per tile.

Everything here assumes the repo root is your working directory and that it is also
`LLM_DK_DIR` (the env scripts set this to `$(pwd)`).

---

## 1. Install Miniforge / conda

If you don't already have conda:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# answer "yes" to conda init, then log out/in
conda env list   # verify
```

## 2. Get the code

```bash
git clone <this-repo> llm-devkit-pvc
cd llm-devkit-pvc

# Upstream training code — checkout the validated commit for reproducibility.
git clone https://github.com/deepspeedai/Megatron-DeepSpeed.git
git -C Megatron-DeepSpeed checkout aab2f31
```

## 3. Create the conda env

```bash
conda create -y --name gpt_pt28 python=3.9.7
conda activate gpt_pt28
```

## 4. Build the stack

```bash
source env/build.sh
```

This installs the pinned stack (see the table in the top-level README):
`torch==2.8.0+xpu`, `intel-extension-for-pytorch==2.8.10+xpu`,
`oneccl_bind_pt==2.8.0+xpu`, `deepspeed==0.16.9`, plus `requirements.txt`.

Verify:

```bash
python -c "import torch, intel_extension_for_pytorch as ipex; \
print(torch.__version__, ipex.__version__); print(torch.xpu.is_available(), torch.xpu.device_count())"
```

You should see the pinned versions and `True 8` on a full node.

> **New conda env?** Remove `~/.cache/torch_extensions` first, otherwise ninja will not
> rebuild the DeepSpeed ops for the new environment.

## 5. Get the dataset

See [DATASET.md](DATASET.md). You need `gpt2-vocab.json`, `gpt2-merges.txt`, and the
preprocessed `BookCorpusDataset_text_document.{bin,idx}` under
`Megatron-DeepSpeed/dataset/`.

## 6. Allocate a node

```bash
salloc -N 1 --account=<YOUR_SLURM_ACCOUNT> -p <PARTITION> -t 01:00:00
# site-specific extras you may need: --qos=<QOS>, --reservation=<NAME>
```

On the allocated node:

```bash
conda activate gpt_pt28
cd <path>/llm-devkit-pvc
source env/set_env.sh          # sets ZE_AFFINITY_MASK + oneCCL/Level-Zero vars
```

Edit the MPI `module load` line in `env/set_env.sh` to match your site — the repo ships
it commented out because module names are cluster-specific. See [TUNING.md](TUNING.md)
for the env vars.

## 7. Generate hostfiles

```bash
cd examples
source generate_hostfile.sh    # writes hostfile_deepspeed + hostfile_mpich
cd ..
```

## 8. 3.6B warm-up (single node)

```bash
cd Megatron-DeepSpeed
bash ../examples/run_3.6b.sh
```

Watch for the per-iteration log lines (`iteration 1/…`). Reaching iteration 1 confirms
the stack, dataset, and communication backend are all working. The initialization phase
takes a while; by default the run trains for 10 iterations and may print a benign error
on exit that can be ignored. Logs land under `Megatron-DeepSpeed/logs/…/output.log`.

## 9. 20B scale-out (multi-node)

20B exceeds a single tile's 64 GB, so it needs model parallelism — you can't run it with
data parallelism alone. The preset targets **4 nodes** (`WORLD_SIZE=32`) with `TP=8`
(kept within one node), `PP=4`, and ZeRO-1. Allocate the nodes, then repeat steps 6–7 so
the hostfiles cover every node:

```bash
salloc -N 4 --account=<YOUR_SLURM_ACCOUNT> -p <PARTITION> -t 01:00:00
# on an allocated node:
conda activate gpt_pt28
cd <path>/llm-devkit-pvc && source env/set_env.sh
cd examples && source generate_hostfile.sh && cd ../Megatron-DeepSpeed

# validated default: TP=8, PP=4, ZeRO-1 on 4 nodes (WORLD_SIZE = nodes × 8)
WORLD_SIZE=32 bash ../examples/run_20b.sh

# scale out with data-parallel replicas — 8 nodes gives DP=2
WORLD_SIZE=64 bash ../examples/run_20b.sh
```

## 10. 175B scale-out (large multi-node)

The 175B preset (`NLAYERS=96`, `HIDDEN=12288`, `HEADS=96`) uses an empirically tuned
config for the 1550: `TP=8`, `PP=16`, `MICRO_BATCH=3`, `GRAD_ACC_STEPS=100`, ZeRO-1.
One model-parallel replica spans `TP × PP = 128` tiles = **16 nodes**; scale out by adding
data-parallel replicas. Allocate the nodes, then repeat steps 6–7 so the hostfiles cover
every node:

```bash
salloc -N 16 --account=<YOUR_SLURM_ACCOUNT> -p <PARTITION> -t 01:00:00
# on an allocated node:
conda activate gpt_pt28
cd <path>/llm-devkit-pvc && source env/set_env.sh
cd examples && source generate_hostfile.sh && cd ../Megatron-DeepSpeed

# validated default: one replica on 16 nodes (WORLD_SIZE=128)
bash ../examples/run_175b.sh

# scale out with data-parallel replicas — 32 nodes gives DP=2
WORLD_SIZE=256 bash ../examples/run_175b.sh
```

## Batch (SBATCH) variant

Instead of interactive `salloc`, use the batch submission layer:

```bash
cp slurm/site.conf.example slurm/site.conf   # fill in for your site, once
slurm/submit.sh --model 3.6b --world-size 8 --tp 1 --pp 1 \
    --micro-batch 8 --grad-acc 1 --zero 2
```

This validates the config and submits through `slurm/jobfile.sh` — the same path
`slurm/sweep.sh` (CSV-driven sweeps) and `slurm/scaling.sh` (175B scaling studies) use. See
[BATCH.md](BATCH.md) for the full submit/sweep/scale workflow.

`examples/submit_3.6b_slurm.sh` still exists as a minimal fallback: a single SBATCH file you edit
directly, with no `slurm/site.conf` to set up first. Prefer `slurm/submit.sh` above for anything
beyond a one-off — it's the only path that validates the config before submitting.

---

## Tuning the parallelism

The presets export a handful of variables that `gpt.sh` turns into Megatron + DeepSpeed
flags:

| Var | Meaning | Constraint |
|---|---|---|
| `WORLD_SIZE` | total ranks (= tiles) | `= nodes × 8` |
| `TP` | tensor-model-parallel size | `TP ≤ 8` (tiles/node) |
| `PP` | pipeline-model-parallel size | `PP` divides `NLAYERS` |
| `ZERO_STAGE` | DeepSpeed ZeRO stage (1/2/3) | use 1 for scale-out (see below) |
| `MICRO_BATCH` | micro-batch per rank | — |
| `GRAD_ACC_STEPS` | gradient-accumulation steps | — |
| `NLAYERS` / `HIDDEN` / `HEADS` / `SEQ` | model shape | `HIDDEN % HEADS == 0` |

Derived: `GLOBAL_BATCH = WORLD_SIZE × MICRO_BATCH × GRAD_ACC_STEPS / TP / PP`.

The data-parallel degree is `WORLD_SIZE / (TP × PP)` — it must be a positive integer, so
choose `TP` and `PP` that divide `WORLD_SIZE`.

**Choosing the parallelism.** The strategy that maximizes throughput on this hardware
(and the general recipe for these models) is:

1. **Keep tensor parallelism within one node — `TP ≤ 8`.** TP has the heaviest
   communication (an all-reduce per layer); crossing the node boundary onto the slower
   inter-node fabric collapses throughput. On PVC that means `TP` up to the 8 tiles/node.
2. **Use pipeline parallelism to fit the rest of the model**, with enough micro-batches
   (`GRAD_ACC_STEPS`) to keep the pipeline full and the bubble small. `PP` must divide
   `NLAYERS`.
3. **Scale out with data-parallel replicas at `ZERO_STAGE=1`.** ZeRO-1 shards only the
   optimizer state, which is a one-time reduce-scatter/all-gather per step. ZeRO-2 and
   ZeRO-3 additionally shard gradients and parameters, adding all-gathers on the critical
   path whose cost dominates over the inter-node fabric — so **prefer ZeRO-1 for
   multi-node scale-out**, and reach for higher stages only single-node or when memory
   genuinely forces it.

## Troubleshooting

- **`torch.xpu.is_available()` is False** — the XPU runtime/driver isn't visible; check
  your GPU driver modules and that you're on a compute node, not the login node.
- **ninja rebuilds every run / stale ops** — clear `~/.cache/torch_extensions`.
- **Hangs at startup on multi-node** — hostfiles don't cover all nodes, or the launcher
  can't reach oneCCL/MPI transport. Regenerate hostfiles and see [TUNING.md](TUNING.md)
  for `CCL_ATL_TRANSPORT` / `CCL_PROCESS_LAUNCHER`.
- **OOM** — lower `MICRO_BATCH`, or raise `PP` (more pipeline stages = less per-tile
  memory). Raising `ZERO_STAGE` also helps but hurts multi-node throughput; prefer more
  `PP` for scale-out.
