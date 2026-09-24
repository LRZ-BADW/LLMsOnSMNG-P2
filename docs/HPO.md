# Hyperparameter search (175B)

`tools/hpo/` runs a [DeepHyper](https://deephyper.readthedocs.io/) Bayesian search over the
175B parallelism knobs — `PP`, `TP`, `MICRO_BATCH`, `GRAD_ACC_STEPS` — to find the configuration
that maximizes achieved TFLOPs. It submits through the same `slurm/submit.sh` the
[batch layer](BATCH.md) uses, so it shares that layer's validation and needs the same
`slurm/site.conf` set up first.

## Install

```bash
pip install -r tools/hpo/requirements-hpo.txt
```

Just `deephyper`, kept out of the main `requirements.txt` since it's only needed for this search.

## The search space

There's no external space file — the space is defined inline in
`tools/hpo/runHyperSearch-175b.py`:

```python
problem.add_hyperparameter([12, 16, 24, 32, 48, 96], "PP")
problem.add_hyperparameter([1, 2, 4, 8], "TP")
problem.add_hyperparameter([1, 2, 4, 6, 8, 10], "MBS")
problem.add_hyperparameter([1, 10, 25, 50, 75, 100], "GAS")
```

`ZeRO` is fixed at `1` in `run()`, not part of the search — that's a deliberate carry-over from
the source this was ported from, not an oversight. To change the space, edit these lines directly.

## Run it

```bash
cd tools/hpo
python runHyperSearch-175b.py
```

`num_workers` (currently `5`, set in the `if __name__ == "__main__":` block) is **the number of
SLURM jobs queued concurrently**, not a local process count. This is why the concurrency-safety
work on the batch harness (per-job hostfiles, per-job DeepSpeed config filenames, a pinned
rendezvous port) is a hard prerequisite for this search — without it, concurrent trials clobber
each other's hostfiles and launch configs.

Each trial computes `WORLD_SIZE = TP * PP` and calls `slurm/submit.sh` with it. A handful of
`TP`/`PP` combinations in the space above produce a `WORLD_SIZE` that isn't a multiple of
`TILES_PER_NODE`; `submit.sh` rejects those at submission time the same as it would for a manually
submitted job. That trial isn't retried or pre-filtered out of the space — it's reported to
DeepHyper as a failed evaluation (see below).

## Reading results

`CBO`'s `log_dir` is `${RUNS_DIR}/hpo_175b_<timestamp>/`, containing DeepHyper's own
`results.csv` — one row per trial, with the search's hyperparameter columns plus the objective
value `run()` returned.

Some rows may show `"F_error"` instead of a TFLOPs number. That's DeepHyper's failed-evaluation
sentinel, not a bug in the search — it means either `submit.sh` rejected the config outright (see
above), or the training job ran but never reached the iteration `get_tflops()` looks for
(`TRAIN_ITER = 5` in the script). Both are normal outcomes for a search that's deliberately
exploring the edges of the space.
