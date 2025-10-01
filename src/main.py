"""src/main.py
Orchestrator that
    1) reads a YAML list of experiment variations,
    2) launches `python -m src.train` in separate subprocesses (1 GPU each),
    3) tees stdout/stderr into per-run log files **and** main stdout,
    4) triggers evaluation once all runs have finished.
"""
from __future__ import annotations

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
import threading
import queue
import yaml
import json


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
TRAIN_MODULE = "src.train"
EVALUATE_MODULE = "src.evaluate"


def _tee_stream(stream, log_file_path: Path, prefix: str = ""):
    """Read from *stream*, write to both stdout and file concurrently."""
    with open(log_file_path, "wb") as log_f:
        for line in iter(stream.readline, b""):
            sys.stdout.buffer.write(prefix.encode() + line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()


def _launch_subprocess(cmd: list[str], env: dict[str, str], stdout_path: Path, stderr_path: Path):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # Tee stdout and stderr in background threads
    threading.Thread(target=_tee_stream, args=(proc.stdout, stdout_path), daemon=True).start()
    threading.Thread(target=_tee_stream, args=(proc.stderr, stderr_path, "[stderr] "), daemon=True).start()
    return proc


def _available_gpus() -> list[int]:
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except ImportError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Run all experiments defined in a YAML file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true")
    group.add_argument("--full-experiment", action="store_true")
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / "config" / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    experiments = cfg["experiments"]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    gpus = _available_gpus()
    print(f"==== Main Orchestrator ====")
    print(json.dumps({"num_experiments": len(experiments), "available_gpus": gpus}))

    # Queue to manage waiting runs if GPUs are insufficient
    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    gpu_queue_empty_placeholder = -1  # CPU fallback

    running_procs = []  # list[tuple[subprocess.Popen, int]]

    def _start_run(exp_cfg):
        gpu_id = gpu_queue.get() if not gpu_queue.empty() else gpu_queue_empty_placeholder
        run_id = exp_cfg["run_id"]
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_file = run_dir / "config.yaml"
        with open(cfg_file, "w") as f:
            yaml.safe_dump(exp_cfg, f)

        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        cmd = [sys.executable, "-m", TRAIN_MODULE,
               "--config", str(cfg_file),
               "--results-dir", str(run_dir)]
        proc = _launch_subprocess(cmd, env, stdout_path, stderr_path)
        running_procs.append((proc, gpu_id))

    # Initially start as many as GPUs
    for exp in experiments[:len(gpus) if gpus else 1]:
        _start_run(exp)

    remaining = experiments[len(running_procs):]

    # Polling loop
    while running_procs or remaining:
        time.sleep(5)
        # Check finished procs
        for proc, gpu_id in running_procs.copy():
            if proc.poll() is not None:  # finished
                running_procs.remove((proc, gpu_id))
                if gpu_id >= 0:
                    gpu_queue.put(gpu_id)  # free GPU
        # Launch new if resources free
        while remaining and (not gpu_queue.empty() or not gpus):
            _start_run(remaining.pop(0))

    # -------- evaluation --------
    eval_cmd = [sys.executable, "-m", EVALUATE_MODULE, "--results-dir", str(results_dir)]
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
