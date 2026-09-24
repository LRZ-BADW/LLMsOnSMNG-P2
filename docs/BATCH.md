# Batch submission

`slurm/` submits, sweeps, and scales training jobs without touching a SBATCH template by hand.
Every script here shares one submission path (`slurm/submit.sh`) and one site config
(`slurm/site.conf`), so a config that's valid once is valid everywhere it's reused.

## Set up `site.conf`

```bash
cp slurm/site.conf.example slurm/site.conf
# edit slurm/site.conf: SLURM_ACCOUNT, PARTITION, CONDA_ENV, REPO_DIR, RUNS_DIR, ...
```

This is the one gitignored file every site-specific value routes through — account, partition,
conda env, repo path, output directory, checkpoint root. No script under `slurm/` or `tools/hpo/`
hardcodes a site value directly; if you ever find one that does, that script has a bug.

## Submit one job

```bash
slurm/submit.sh --model 20b --world-size 32 --tp 8 --pp 4 \
    --micro-batch 1 --grad-acc 8 --zero 1
```

Required: `--model` (`3.6b`, `20b`, or `175b`), `--world-size`, `--tp`, `--pp`, `--micro-batch`,
`--grad-acc`, `--zero`. Optional: `--time` (defaults to `DEFAULT_TIME` from `site.conf`),
`--train-iter` (default `10`), `--dtype` (default `bf16`), `--dry-run`.

Before calling `sbatch`, `submit.sh` validates the config against the constraints in the
[README's parallelism table](../README.md#parallelism-model):

- `TP <= TILES_PER_NODE` — tensor parallelism must fit on one node.
- `PP` divides the model's `NLAYERS` (30 for 3.6B, 44 for 20B, 96 for 175B).
- `WORLD_SIZE % TILES_PER_NODE == 0` — node count must be a whole number.

Any violation exits with an explanation before anything is submitted. Node count is derived as
`WORLD_SIZE / TILES_PER_NODE`, and the partition is `PARTITION_LARGE` instead of `PARTITION` once
node count reaches `LARGE_NODE_THRESHOLD` (if set).

Use `--dry-run` to print the assembled `sbatch` command without submitting it — useful for
checking a config before it ever touches the queue.

## Sweep from a CSV

```bash
slurm/sweep.sh path/to/sweep.csv [--dry-run]
```

Submits one job per row via `submit.sh`, so every row gets the same validation. The CSV needs this
exact header:

```csv
MODEL,PP,TP,MICRO_BATCH,GRAD_ACC_STEPS,ZERO_STAGE,WORLD_SIZE
20b,4,8,1,8,1,32
20b,2,8,1,8,1,16
175b,16,8,3,100,1,128
```

`--dry-run`, `--time`, `--train-iter`, and `--dtype` are forwarded to every row's `submit.sh` call.

## Run a scaling study (175B)

```bash
slurm/scaling.sh --mode strong --factors "1 2 4 8"
```

Scales the 175B base config (`WORLD_SIZE=128 TP=8 PP=16 MICRO_BATCH=3`) by each factor in
`--factors`, submitting one job per factor via `submit.sh`. Two modes:

- **`--mode weak`** — `GRAD_ACC_STEPS` stays fixed at `100` for every factor; total batch grows
  with node count.
- **`--mode strong`** — the global batch stays fixed at `2520`; `GRAD_ACC_STEPS` is derived
  backward from it, shrinking as node count grows. (This is the mode the private
  `1tStrongScaling.sh` got wrong — it never recomputed `GRAD_ACC_STEPS`, so it silently ran weak
  scaling under a "strong" label.)

Both modes were verified against real measured values for factors 1–8 (128 through 1024 tiles).

## Where outputs land

Every job's SLURM output goes to `${RUNS_DIR}/<job_name>-<job_id>.out`, where
`job_name = <model>-tp<TP>-pp<PP>-ws<WORLD_SIZE>`. There's no per-job subdirectory — all logs from
all submission paths (`submit.sh`, `sweep.sh`, `scaling.sh`, the HPO search) land flat in
`RUNS_DIR`.

## The `params.csv` contract

Each `examples/run_*.sh` preset, when `RUNS_DIR` is set, appends a row to
`${RUNS_DIR}/${MODEL}-${SLURM_JOB_ID}-params.csv`:

```
JOB_ID,PP,TP,WORLD_SIZE,MICRO_BATCH,GRAD_ACC_STEPS,GLOBAL_BATCH,ZERO_STAGE,TRAIN_ITER
```

This is how you attribute a log file back to the exact config that produced it: the `JOB_ID`
column is the same `<job_id>` in the log's filename. Without it, a directory full of `.out` files
sweeps produce is just noise — you'd have throughput numbers with no way to know which
`TP`/`PP`/`ZeRO` combination they came from.
