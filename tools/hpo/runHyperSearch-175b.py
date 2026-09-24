"""DeepHyper Bayesian search over the 175B parallelism knobs (PP, TP,
MICRO_BATCH, GRAD_ACC_STEPS). Ported from the private
experiments/175b/runHyperSearch-175b.py -- job submission goes through
slurm/submit.sh instead of an inlined sbatch string, so this shares the batch
layer's submission path and validation (TP <= TILES_PER_NODE, PP divides
NLAYERS, WORLD_SIZE % TILES_PER_NODE == 0). ZeRO is fixed at 1 below, not
searched, matching the private script.
"""
import datetime
import subprocess
import sys
from pathlib import Path

from deephyper.evaluator import Evaluator
from deephyper.hpo import CBO, HpProblem

from slurm_utils import submit_and_monitor_job, get_tflops

MODEL = "175b"
TRAIN_ITER = 5
POLL_INTERVAL = 120

REPO_DIR = Path(__file__).resolve().parents[2]
SUBMIT_SH = REPO_DIR / "slurm" / "submit.sh"
SITE_CONF = REPO_DIR / "slurm" / "site.conf"


def site_conf_var(name):
    """Read one variable out of slurm/site.conf by sourcing it in a subshell,
    the same single indirection submit.sh/sweep.sh rely on."""
    if not SITE_CONF.exists():
        print(f"Missing {SITE_CONF} -- copy slurm/site.conf.example and fill it in.", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(
        ["bash", "-c", f'set -eu; . "{SITE_CONF}" >/dev/null; echo "${{{name}}}"'],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


RUNS_DIR = site_conf_var("RUNS_DIR")


def run(config):
    PP = config["PP"]
    TP = config["TP"]
    MBS = config["MBS"]
    GAS = config["GAS"]
    ZeRO = 1  # fixed, not searched -- matches the private hardcode

    world_size = TP * PP
    job_name = f"{MODEL}-tp{TP}-pp{PP}-ws{world_size}"

    job_command = (
        f"{SUBMIT_SH} --model {MODEL} --world-size {world_size} "
        f"--tp {TP} --pp {PP} --micro-batch {MBS} --grad-acc {GAS} "
        f"--zero {ZeRO} --train-iter {TRAIN_ITER}"
    )
    result = submit_and_monitor_job(job_command, job_name, RUNS_DIR, POLL_INTERVAL)
    if result is None:
        # submit.sh rejected the config (e.g. WORLD_SIZE not a multiple of
        # TILES_PER_NODE) -- a failed evaluation, not a crash.
        return "F_error"

    return get_tflops(result["output"], TRAIN_ITER)


problem = HpProblem()
problem.add_hyperparameter([12, 16, 24, 32, 48, 96], "PP")     # 6 possible values
problem.add_hyperparameter([1, 2, 4, 8], "TP")                 # 4 possible values
problem.add_hyperparameter([1, 2, 4, 6, 8, 10], "MBS")         # 6 possible values
problem.add_hyperparameter([1, 10, 25, 50, 75, 100], "GAS")    # 6 possible values

if __name__ == "__main__":
    num_workers = 5
    max_evals = 200  # 0.5 * possible combinations = 0.5*6*4*6*6 = 432
    method = "process"

    evaluator = Evaluator.create(
        run,
        method=method,
        method_kwargs={"num_workers": num_workers},
    )
    search = CBO(
        problem,
        evaluator,
        random_state=42,
        log_dir=f"{RUNS_DIR}/hpo_{MODEL}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/",
    )
    results = search.search(max_evals=max_evals)
