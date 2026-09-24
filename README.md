# llm-devkit-pvc

Reference recipe for **GPT pre-training on the Intel Data Center GPU Max 1550
(Ponte Vecchio / PVC)** using
[Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed) with
ZeRO + tensor + pipeline parallelism, on a SLURM cluster of OAM nodes (8 tiles/node).

The Max 1550 is the dual-tile 128 GB part (2 × 64 GB HBM per card), so an OAM node of
4 cards presents **8 tiles × 64 GB**. The node topology, memory budgets, and the
validated parallelism configs below are all specific to the 1550 — the single-tile
Max 1100 has a different layout and these presets do not carry over unchanged.

This is a **minimal, self-contained toolkit**: pinned Python stack, environment setup,
the DeepSpeed launch harness, and a SLURM submission template — plus expert docs that
explain the parallelism knobs and the oneCCL / Level-Zero tuning that matter on PVC.

> Validated stack: **PyTorch 2.8 + Intel Extension for PyTorch 2.8.10 + oneCCL 2.8 + DeepSpeed 0.16.9**.

---

## Stack

| Component | Pin |
|---|---|
| Python | 3.9.7 |
| PyTorch (XPU) | `torch==2.8.0` (`--index-url https://download.pytorch.org/whl/xpu`) |
| torchvision / torchaudio | `0.23.0` / `2.8.0` |
| Intel Extension for PyTorch | `intel-extension-for-pytorch==2.8.10+xpu` |
| oneCCL bindings for PyTorch | `oneccl_bind_pt==2.8.0+xpu` |
| DeepSpeed | `deepspeed==0.16.9` |
| transformers | `4.51.3` |
| Megatron-DeepSpeed | commit `aab2f31` of `github.com/deepspeedai/Megatron-DeepSpeed` |

On the 2.8 stack the accelerator is selected implicitly via IPEX — you do **not** need to
set `DS_ACCELERATOR=xpu` manually.

> **Newer toolchain (forward-looking).** This repo pins the PyTorch 2.8 + IPEX stack
> that the reference experiments used. An IPEX-free variant on **PyTorch 2.12 XPU**
> (where the wheel bundles oneCCL and the Level-Zero/SYCL runtime, so neither IPEX nor a
> separate `oneccl_bind_pt` is needed) has also been validated end-to-end on PVC Max 1550
> OAM nodes, at 3.6B (single-node) and 20B (4-node) scale. It requires a recent GPU driver
> and a few adjustments — `DS_ACCELERATOR=xpu` set explicitly (no IPEX to auto-select the
> device), `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, and a minor DeepSpeed-config tweak for
> DeepSpeed ≥ 0.19. The pinned 2.8 stack remains the recommended, fully-reproducible
> baseline.

---

## Repo map

```
llm-devkit-pvc/
├── requirements.txt          # pinned pure-python deps
├── env/
│   ├── build.sh              # create the XPU stack in the active conda env
│   └── set_env.sh            # runtime env: ZE_AFFINITY_MASK, oneCCL, Level-Zero
├── examples/
│   ├── gpt.sh                # builds the deepspeed launch command
│   ├── generate_config.sh    # emits the DeepSpeed ZeRO JSON for the chosen stage
│   ├── generate_hostfile.sh  # SLURM nodelist -> deepspeed / mpich hostfiles
│   ├── run_3.6b.sh           # 3.6B model preset
│   ├── run_20b.sh            # 20B model preset
│   ├── run_175b.sh           # 175B model preset (large scale-out)
│   └── submit_3.6b_slurm.sh  # minimal single-job SBATCH example, no site.conf needed
├── slurm/
│   ├── site.conf.example  # site values every script below routes through
│   ├── jobfile.sh         # generic SBATCH jobfile (called by submit.sh, not sbatch'd directly)
│   ├── submit.sh          # submit one job, with config validation
│   ├── sweep.sh           # submit one job per row of a CSV
│   └── scaling.sh         # 175B weak/strong scaling sweep
├── tools/hpo/
│   ├── slurm_utils.py          # SLURM submit/monitor + TFLOPs log parsing
│   ├── runHyperSearch-175b.py  # DeepHyper search over the 175B parallelism knobs
│   └── requirements-hpo.txt    # deephyper, kept out of the main requirements.txt
└── docs/
    ├── SETUP.md              # full step-by-step recipe
    ├── TUNING.md             # oneCCL / SYCL / Level-Zero env-var reference
    ├── DATASET.md            # fetch & preprocess the training corpus
    ├── BATCH.md              # submit / sweep / scale training jobs
    └── HPO.md                # DeepHyper search over the parallelism knobs
```

---

## 5-minute quickstart

```bash
# 0. clone this repo and the training code
git clone <this-repo> llm-devkit-pvc
cd llm-devkit-pvc
git clone https://github.com/deepspeedai/Megatron-DeepSpeed.git
git -C Megatron-DeepSpeed checkout aab2f31   # validated commit

# 1. create + activate a conda env
conda create -y --name gpt_pt28 python=3.9.7
conda activate gpt_pt28

# 2. build the XPU stack into the env
source env/build.sh

# 3. get one node (8 tiles) from SLURM
salloc -N 1 --account=<YOUR_SLURM_ACCOUNT> -p <PARTITION> -t 01:00:00
# (on the allocated node)
conda activate gpt_pt28
source env/set_env.sh

# 4. generate hostfiles and run the 3.6B warm-up
cd examples && source generate_hostfile.sh
cd ../Megatron-DeepSpeed
bash ../examples/run_3.6b.sh
```

See **[docs/SETUP.md](docs/SETUP.md)** for the full recipe (single-node → 4-node 20B
scale-out and the batch-job variant), **[docs/TUNING.md](docs/TUNING.md)** for the PVC
communication tuning, and **[docs/DATASET.md](docs/DATASET.md)** for the corpus.

---

## Running at scale

Beyond one interactive job, `slurm/` and `tools/hpo/` add a batch-submission layer and a
DeepHyper hyperparameter search, both driven off a single site config so nothing site-specific
is hardcoded into a script:

- **[docs/BATCH.md](docs/BATCH.md)** — submit one job, sweep a CSV of configs, or run a 175B
  weak/strong scaling study, all through `slurm/submit.sh`.
- **[docs/HPO.md](docs/HPO.md)** — a Bayesian search over the 175B parallelism knobs
  (`tools/hpo/runHyperSearch-175b.py`), submitting through that same path.

---

## Parallelism model

Training is configured through a small set of environment variables consumed by the
`examples/*.sh` presets:

| Var | Meaning | Constraint |
|---|---|---|
| `WORLD_SIZE` | total number of ranks (= tiles) | `= nodes × 8` |
| `TP` | tensor-model-parallel size | `TP ≤ tiles per node` (≤ 8) |
| `PP` | pipeline-model-parallel size | `PP` must divide `NLAYERS` |
| `ZERO_STAGE` | DeepSpeed ZeRO stage (1/2/3) | `1` for multi-node scale-out |
| `MICRO_BATCH` | micro-batch per rank | — |
| `GRAD_ACC_STEPS` | gradient-accumulation steps | — |

The global batch is derived: `GLOBAL_BATCH = WORLD_SIZE × MICRO_BATCH × GRAD_ACC_STEPS / TP / PP`.

The validated strategy: keep **tensor parallelism inside one node** (`TP ≤ 8`), use
**pipeline parallelism** with enough gradient accumulation to fill the pipeline, and
**scale out with data-parallel replicas at ZeRO-1**. Higher ZeRO stages add
gradient/parameter collectives that dominate over the inter-node fabric, so they hurt
multi-node throughput — see [docs/SETUP.md](docs/SETUP.md#tuning-the-parallelism). The
presets ship with these defaults: 20B = `TP=8, PP=4` (4 nodes); 175B = `TP=8, PP=16`
(16 nodes).

---

## License

[Apache-2.0](LICENSE). This toolkit wraps and pins upstream open-source projects
(Megatron-DeepSpeed, DeepSpeed, Intel Extension for PyTorch); those retain their own licenses.

---

## Authors

- Ajay Navilarekal Rajgopal (LRZ, [@ajaynr](https://github.com/ajaynr))
- Nikolai Solmsdorf (Intel, [@nsolmsdo](https://github.com/nsolmsdo))
