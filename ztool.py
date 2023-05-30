# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
This file is part of the Zeus deep learning library.
    ztool.py provides a set of tools you can use to manage checkpoints, protect GPUs, ...
"""
import zeus
import os
import copy
import shutil
import argparse
from pycg import exp, wdb
from pathlib import Path


def run_clean(opt):
    import wandb

    with open(".premature_checkpoints", "r") as f:
        file_paths = f.readlines()

    file_paths = [t.strip() for t in file_paths]

    # Parse and keep only remaining paths
    remaining_paths = []
    for fpath in file_paths:
        if fpath.startswith("wdb:"):
            run_path = fpath.split(":")[1]
            try:
                wandb_run = wandb.Api().run(run_path)
            except wandb.errors.CommError:
                continue
            # I assume no wandb ckpt exists, but this is not true.
            remaining_paths.append((fpath, False, "wandb"))

        elif Path(fpath).exists() and (Path(fpath) / "hparams.yaml").exists():
            has_ckpt = (Path(fpath) / "checkpoints" / "last.ckpt").exists()
            remaining_paths.append((fpath, has_ckpt, "tb"))

    for pid, (path, has_ckpt, ckpt_type) in enumerate(remaining_paths):
        print(f"{pid}. {path}. \t{'WITH CKPT' if has_ckpt else 'NO CKPT'}")
    user_input = input("Type number (space as sep.) to delete. 'All' to delete everything. >> ")

    removed_paths = []
    if user_input == 'All':
        removed_paths = copy.deepcopy(remaining_paths)
    else:
        idxs = [int(t) for t in user_input.split()]
        removed_paths = [remaining_paths[t] for t in idxs]

    for p in removed_paths:
        exp.logger.info(f"Now Removing {p[0]}...")
        if p[2] == 'tb':
            shutil.rmtree(p[0])
        else:
            run_path = p[0].split(":")[1]
            wandb_run = wandb.Api().run(run_path)
            wandb_run.delete()
        remaining_paths.remove(p)

    with open(".premature_checkpoints", "w") as f:
        for p in remaining_paths:
            f.write(f"{p[0]}\n")

    exp.logger.info("Done!")


def run_upload(opt):
    import wandb

    def get_wildcard_run_names(pattern):
        if pattern.startswith("wdb:"):
            pattern = pattern[4:]

        if ":" in pattern:
            pattern, ckpt_name = pattern.split(":")
        else:
            ckpt_name = None

        pattern = pattern.split("/")
        user_name = pattern[0]
        project_name = pattern[1]
        other_name = "/".join(pattern[2:])

        all_runs = wandb.Api().runs(
            f"{user_name}/{project_name}",
            filters={"display_name": {"$regex": other_name}}
        )

        print("Will upload the following runs:")
        run_names = []
        for rid, run in enumerate(all_runs):
            print(f"   {rid}: {run.name} \t ({run.state})")
            run_names.append('/'.join(run.path) + (f":{ckpt_name}" if ckpt_name is not None else ""))

        return run_names

    # To make rsync work properly, we have to chdir to project root.
    os.chdir(zeus.config.wandb.base)

    # For run_name with wildcard, select everything here:
    if '*' in opt.run_name:
        all_run_names = get_wildcard_run_names(opt.run_name)
    else:
        all_run_names = [opt.run_name]

    # Get run and ckpt info
    ckpt_to_upload = []
    for run_name in all_run_names:
        wdb_run, ckpt_path = wdb.get_wandb_run(run_name, ".", default_ckpt="all")
        ckpt_to_upload.append(ckpt_path)

    # Perform upload
    for ckpt_path in ckpt_to_upload:
        if not ckpt_path.exists():
            print("Warning:", ckpt_path, "does not exist!")
        os.system(f"rsync -a --progress --update --no-owner --no-group --relative {ckpt_path} {zeus.config.wandb.upload}")


def run_protect(opt):
    import torch
    import time
    from threading import Thread, Event
    from pycg.exp import natural_time, get_gpu_status

    class ProjectThread(Thread):
        def __init__(self, gpu_idx, stop_event):
            super().__init__()
            self.stop_event = stop_event
            self.device = torch.device(f"cuda:{gpu_idx}")
            self.run_time = 0.0

            dev_prop = get_gpu_status("localhost", use_nvml=True)[gpu_idx]
            if dev_prop.gpu_mem_usage > 0.5:
                exp.logger.warning(f"GPU idx {gpu_idx} may already in use!")
            protect_bytes = (dev_prop.gpu_mem_total - dev_prop.gpu_mem_byte) * 0.75 / 4
            self.protect_bytes = int(protect_bytes)

        def run(self):
            res = torch.zeros((self.protect_bytes, ), device=self.device)
            start_time = time.time()
            while True:
                res.add_(1.0)
                time.sleep(0.01)
                if self.stop_event.is_set():
                    del res
                    torch.cuda.empty_cache()
                    break
                self.run_time = time.time() - start_time

    protect_threads = [None for _ in range(10)]
    stop_events = [None for _ in range(10)]

    while True:
        prompt = input("(protector) ")
        if prompt == "p":
            res = get_gpu_status("localhost", use_nvml=True)
            for gpu_idx, gpu_info in enumerate(res):
                if protect_threads[gpu_idx] is not None:
                    print(f"[PROTECTING {natural_time(protect_threads[gpu_idx].run_time)}] {gpu_info}")
                else:
                    print(gpu_info)
        elif prompt.startswith("s"):
            target_gpu_idx = prompt.split()[-1]
            target_gpu_idx = [int(t) for t in target_gpu_idx if protect_threads[int(t)] is None]
            target_gpu_idx = set(target_gpu_idx)
            for gpu_idx in target_gpu_idx:
                stop_events[gpu_idx] = Event()
                protect_threads[gpu_idx] = ProjectThread(gpu_idx, stop_events[gpu_idx])
                protect_threads[gpu_idx].start()
        elif prompt.startswith("e"):
            target_gpu_idx = prompt.split()[-1]
            target_gpu_idx = [int(t) for t in target_gpu_idx if protect_threads[int(t)] is not None]
            target_gpu_idx = set(target_gpu_idx)
            for gpu_idx in target_gpu_idx:
                stop_events[gpu_idx].set()
                protect_threads[gpu_idx].join()
                protect_threads[gpu_idx] = stop_events[gpu_idx] = None
        elif prompt.startswith("q"):
            break
        else:
            print("Command not supported.")

    print("exiting...")
    _ = [s.set() for s in stop_events if s is not None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zeus tools')
    sub_parsers = parser.add_subparsers(dest='command', required=True)

    clean_parser = sub_parsers.add_parser('clean', help='Clean unused checkpoints')

    upload_parser = sub_parsers.add_parser('upload', help='Transfer checkpoints to another machine')
    upload_parser.add_argument('run_name', type=str, help='checkpoint to upload')

    protect_parser = sub_parsers.add_parser('protect', help='Start the GPU protection utility')

    args = parser.parse_args()

    if args.command == 'clean':
        run_clean(args)
    elif args.command == 'upload':
        run_upload(args)
    elif args.command == 'protect':
        run_protect(args)
