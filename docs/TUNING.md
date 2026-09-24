# Tuning — oneCCL / SYCL / Level-Zero on PVC

These are the runtime environment variables that most affect correctness and
throughput of multi-tile / multi-node Megatron-DeepSpeed training on Intel Data
Center GPU Max 1550 (PVC). The recommended baseline is what `env/set_env.sh` exports;
this doc explains each one and the order to tune in.

## Recommended baseline

```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
export CCL_OP_SYNC=1
export CCL_SKIP_SCHEDULER=1
export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
```

## Reference

| Variable | What it affects | When it helps | Trade-off / notes |
|---|---|---|---|
| `ZE_AFFINITY_MASK` | Which tiles a process sees (flat indices `0,1,…`) | Always — pins the visible device set | Set by `env/set_env.sh`; a wrong mask silently under-uses the node |
| `ZE_ENABLE_PCI_ID_DEVICE_ORDER` | Orders devices by PCI id instead of driver default | Makes tile enumeration stable & reproducible across runs/nodes | None in practice; keep on |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | Uses Level-Zero immediate command lists | Lower submission latency; usually a throughput win on PVC | Rarely regresses; test off if you see stalls |
| `ENABLE_SDP_FUSION` | Fused scaled-dot-product attention kernels (IPEX) | Faster + lower-memory attention | Turn off only to isolate a suspected attention numerical issue |
| `CCL_OP_SYNC` | Forces synchronous oneCCL collectives | Stability; deterministic collective timing | Can cost overlap/throughput — the first knob to try toggling for speed once stable |
| `CCL_SKIP_SCHEDULER` | Bypasses the oneCCL scheduler for eligible ops | Reduces per-collective overhead | Validate correctness on your stack before trusting at scale |
| `CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD` | Size cutoff for the medium allgatherv algorithm (`0` disables that path) | Tunes allgatherv for ZeRO comms | Optimal value is size/topology dependent; sweep it |
| `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL` | How in-order barriers are implemented in the L0 backend | Works around barrier/perf issues on some driver versions | Driver-dependent; flip if you see barrier hangs or slowdowns |
| `CCL_ATL_TRANSPORT=mpi` | oneCCL transport layer | Needed when the launcher requires MPI transport | Commented out by default; enable to match your MPI launcher |
| `CCL_PROCESS_LAUNCHER=mpi` | How oneCCL discovers ranks | Pair with `CCL_ATL_TRANSPORT=mpi` under MPICH/Intel MPI | Commented out by default |
| `I_MPI_DEBUG` | Intel MPI verbosity | Raise (e.g. 7) to debug rank/pinning/transport issues | Noisy + slows startup; keep low/unset in production |

## Tuning order

1. **Get it correct and stable first** with the baseline above. Confirm a clean 3.6B
   single-node run reaches several iterations.
2. **Launcher/transport**: if multi-node startup hangs, enable `CCL_ATL_TRANSPORT=mpi`
   and `CCL_PROCESS_LAUNCHER=mpi` to match your MPI launcher, and raise `I_MPI_DEBUG`
   to inspect.
3. **Throughput**: once stable, experiment with `CCL_OP_SYNC=0` to recover
   compute/comm overlap, then sweep `CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD` for your
   message sizes. Re-verify loss curves after each change.
4. **Driver quirks**: if you hit barrier hangs or unexpected slowdowns tied to a driver
   update, toggle `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL` and
   `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`.

Always change one variable at a time and confirm both throughput (TFLOP/s, tokens/s)
and correctness (loss) before keeping it.
