"""SLURM job submission/monitoring and log parsing for the DeepHyper search.

Used by runHyperSearch-175b.py so the submission and parsing logic isn't
inlined into the search script itself.
"""
import subprocess
import time
import re


def submit_and_monitor_job(job_command, job_name, job_outputs_dir, poll_interval=120):
    split_command = job_command.split()
    submit_result = subprocess.run(split_command,
                                    capture_output=True,
                                    text=True)

    if submit_result.returncode != 0:
        print(f"Submission failed: {submit_result.stderr}")
        return None

    # "Submitted batch job 12345" -> "12345"
    job_id = submit_result.stdout.strip().split()[-1]
    print(f"Submitted job ID: {job_id}")

    # Monitor job until completion
    while True:
        squeue_result = subprocess.run(['squeue', '-j', job_id],
                                        capture_output=True,
                                        text=True)

        # If job not found in queue, it's completed (or failed)
        if "Invalid job id specified" in squeue_result.stderr or \
           job_id not in squeue_result.stdout:
            break

        print(f"Job {job_id} still running...")
        time.sleep(poll_interval)

    sacct_result = subprocess.run(['sacct', '-j', job_id, '--format=State', '-n'],
                                   capture_output=True,
                                   text=True)
    final_state = sacct_result.stdout.strip()

    # `.out`, matching slurm/submit.sh's own `--output=${RUNS_DIR}/${job_name}-%j.out`
    # (the private convention was `.txt`, driven by a caller-supplied `-o` flag).
    try:
        with open(f'{job_outputs_dir}/{job_name}-{job_id}.out', 'r') as f:
            job_output = f.read()
    except FileNotFoundError:
        job_output = "Output file not found"

    return {
        'job_id': job_id,
        'final_state': final_state,
        'output': job_output,
    }


def get_tflops(output_text, train_iter):
    # `\s+` rather than a fixed space count: the iteration field is
    # right-justified to a constant total width, so the number of spaces
    # before the digits shrinks as train_iter grows more digits.
    pattern = rf'iteration\s+{train_iter}/\s*\d+.*?TFLOPs:\s*(\d+\.?\d*)'

    match = re.search(pattern, output_text)
    if match:
        return float(match.group(1))
    else:
        return "F_error"
