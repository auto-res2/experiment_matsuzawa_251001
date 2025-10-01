"""src/main.py
Experiment orchestrator.  Reads the YAML config file (smoke-test or full-
experiment), spawns one subprocess *per run variation*, allocates a single GPU
per subprocess (or CPU if GPUs are exhausted), streams logs to both stdout and
run-specific log files, and finally calls evaluate.py to generate comparison
figures.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch

###########################################################################
# --------------------------- Helper functions -------------------------- #
###########################################################################

def available_gpu_ids() -> List[int]:
    """Returns the list of *visible* GPU IDs (empty if no CUDA)."""
    if not torch.cuda.is_available():
        return []
    # Honour CUDA_VISIBLE_DEVICES if it exists
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        # Map *relative* ids (0..n_visible-1) to *absolute* device ordinal
        parts = [int(x) for x in visible.split(",") if x.strip()]
        return list(range(len(parts)))
    # Otherwise use all devices
    return list(range(torch.cuda.device_count()))


def tee_stream(stream, out_file):
    """Tee *byte* stream from subprocess to both stdout/stderr and file."""
    for line in iter(stream.readline, b""):
        decoded = line.decode("utf-8", errors="replace")
        print(decoded, end="", flush=True)  # Pass-through to parent console
        out_file.write(decoded)
        out_file.flush()
    stream.close()

###########################################################################
# ------------------------------ Launcher ------------------------------ #
###########################################################################

def launch_training_subprocess(run_cfg: Dict[str, Any], full_cfg_path: Path, results_dir: Path,
                               gpu_id: int | None, smoke_test: bool):
    """Spawns `python -m src.train ...` with correct environment and I/O redirection."""
    run_id = run_cfg["id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_file = open(run_dir / "stdout.log", "w", encoding="utf-8")
    stderr_file = open(run_dir / "stderr.log", "w", encoding="utf-8")

    cmd = [sys.executable, "-m", "src.train", "--config-path", str(full_cfg_path),
           "--run-id", run_id, "--results-dir", str(results_dir)]
    if smoke_test:
        cmd.append("--smoke-test")

    env = os.environ.copy()
    # Restrict GPU visibility for this subprocess
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    # Threaded tee to duplicate streams
    threading.Thread(target=tee_stream, args=(proc.stdout, stdout_file), daemon=True).start()  # type: ignore[arg-type]
    threading.Thread(target=tee_stream, args=(proc.stderr, stderr_file), daemon=True).start()  # type: ignore[arg-type]

    return proc, stdout_file, stderr_file

###########################################################################
# -------------------------------- Main -------------------------------- #
###########################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Orchestrate all experiment variations")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick CI smoke-test")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment set")
    p.add_argument("--results-dir", type=str, required=True, help="Directory to save all outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1) Load YAML config (smoke-test vs full experiment)
    # ------------------------------------------------------------------ #
    config_path = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        full_cfg = yaml.safe_load(fh)

    experiments: List[Dict[str, Any]] = full_cfg["experiments"]

    # ------------------------------------------------------------------ #
    # 2) Prepare results directory
    # ------------------------------------------------------------------ #
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist a (possibly modified) copy of the configuration for provenance
    with open(results_dir / "used_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(full_cfg, fh)

    # ------------------------------------------------------------------ #
    # 3) GPU scheduling: one GPU per run, queue if needed
    # ------------------------------------------------------------------ #
    idle_gpus = deque(available_gpu_ids())
    pending_runs = deque(experiments)
    active_processes = []  # list of tuples (proc, gpu_id, stdout_fh, stderr_fh)

    while pending_runs or active_processes:
        # Launch as many as we can with currently idle GPUs (or CPU if none)
        while pending_runs and (idle_gpus or not available_gpu_ids()):
            run_cfg = pending_runs.popleft()
            gpu_to_use = idle_gpus.popleft() if idle_gpus else None
            proc, out_fh, err_fh = None, None, None  # just for mypy happiness
            proc, out_fh, err_fh = launch_training_subprocess(run_cfg, full_cfg_path=config_path,
                                                             results_dir=results_dir, gpu_id=gpu_to_use,
                                                             smoke_test=args.smoke_test)
            active_processes.append((proc, gpu_to_use, out_fh, err_fh))
            print(f"[main] Launched run '{run_cfg['id']}' on GPU {gpu_to_use}")
            time.sleep(0.5)  # Small delay to avoid race conditions

        # Poll active processes
        still_active = []
        for proc, gpu_id, out_fh, err_fh in active_processes:
            if proc.poll() is None:
                still_active.append((proc, gpu_id, out_fh, err_fh))
            else:
                # Completed
                out_fh.close()
                err_fh.close()
                if gpu_id is not None:
                    idle_gpus.append(gpu_id)
        active_processes = still_active
        time.sleep(1.0)

    # ------------------------------------------------------------------ #
    # 4) All runs finished – perform evaluation & visualisation
    # ------------------------------------------------------------------ #
    print("[main] All experiment variations completed. Running evaluation…", flush=True)
    subprocess.run([sys.executable, "-m", "src.evaluate", str(results_dir)], check=True)

###########################################################################

if __name__ == "__main__":
    main()
