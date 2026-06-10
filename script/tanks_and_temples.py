# This script is modified from mip-splatting.
# Original project link: https://github.com/autonomousvision/mip-splatting/blob/main/scripts/run_mipnerf360.py
# Copyright belongs to the original authors, see the original project license for details.

import os
import subprocess
import sys
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

scenes = ["Auditorium", "Ballroom", "Barn", "Caterpillar", "Church", "Courthouse", "Courtroom", "Family", "Francis", "Horse", "Ignatius", "Lighthouse", "M60", "Meetingroom", "Museum", "Palace", "Panther", "Playground", "Temple", "Train", "Truck"]
factors = [1] * len(scenes)

excluded_gpus = set([])

output_dir = "./output/tanks_and_temples"

dry_run = False

jobs = list(zip(scenes, factors))


def run_cmd(cmd, gpu_id):
    """Run a command with proper env vars on Windows."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(cmd)
    if not dry_run:
        result = subprocess.run(cmd, shell=True, env=env)
        if result.returncode != 0:
            print(f"Warning: command failed with return code {result.returncode}")


def train_scene(gpu, scene, factor):
    python_exe = sys.executable

    cmd = f"{python_exe} train.py -s tanks_and_temples/{scene} -m {output_dir}/{scene} --eval -r {factor} --port {6009+int(gpu)} --data_device cpu"
    run_cmd(cmd, gpu)

    cmd = f"{python_exe} render.py -m {output_dir}/{scene} --data_device cpu --skip_train"
    run_cmd(cmd, gpu)

    cmd = f"{python_exe} metrics.py -m {output_dir}/{scene}"
    run_cmd(cmd, gpu)
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()

    while jobs or future_to_job:
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)

        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)
            gpu = job[0]
            reserved_gpus.discard(gpu)
            print(f"Job {job} has finished, releasing GPU {gpu}")
        time.sleep(5)

    print("All jobs have been processed.")


with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
