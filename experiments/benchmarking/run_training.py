import time
import os
import sys
import numpy as np
import argparse
import pathlib, shutil
from datetime import datetime
from subprocess import Popen, DEVNULL
from rich import print


def make_code_snap(experiment, slurm_dir='exp_sweep'):
    now = datetime.now()
    snap_dir = pathlib.Path.cwd() / slurm_dir
    snap_dir /= now.strftime('%Y-%m-%d')
    snap_dir /= now.strftime('%H-%M-%S') + f'_{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        search_dir = src_dir / dir
        target_dir = snap_dir / "code" / dir
        ignore_patterns = [
            "*CARL/tmp*",
            "*exp_sweep*",
        ]
        shutil.copytree(search_dir, target_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*ignore_patterns))

    # Copy source code
    dirs_to_copy = [
        'carl',
        'experiments/benchmarking',
        'experiments/context_gating',
        'experiments/carlbench',
        'experiments/common/utils',
        'experiments/evaluation_protocol'
    ]
    src_dir = pathlib.Path.cwd()
    print(src_dir)
    for dir in dirs_to_copy:
        copy_dir(dir)

    # Copy repository / environment infos
    rootfiles = src_dir.glob("*")
    for f in rootfiles:
        if f.is_file() and not str(f.name).startswith("."):
            shutil.copyfile(f, snap_dir / 'code' / f.name)

    # Copy all init files
    initfiles = list(src_dir.rglob("*/__init__.py"))
    initfiles.sort()
    initdirs = dirs_to_copy + [
        "experiments/__init__.py",
        "experiments/common/__init__.py"
    ]
    for f in initfiles:
        if f.is_file() and \
                not str(f.name).startswith(".")\
                and np.any([d in str(f) for d in initdirs])\
                and "exp_sweep" not in str(f):
            fn = str(f)[len(str(src_dir)) + 1:]
            newfn = snap_dir / 'code' / fn
            shutil.copyfile(f, newfn)

    return snap_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true')
    args, unknown_args = parser.parse_known_args()  # unknown args are hydra commands
    # unknown_args = [f"'{a}'" for a in unknown_args]

    experiment = "benchmark_train"
    snap_dir = make_code_snap(experiment)
    print("Snap dir:", str(snap_dir))

    add_multirun_flag = False
    if unknown_args[-1] == "-m":
        unknown_args.pop(-1)
        add_multirun_flag = True

    unknown_args += [
        f"hydra.run.dir={snap_dir}",
        f"hydra.sweep.dir={snap_dir}",
    ]

    if add_multirun_flag:
        unknown_args += ["-m"]

    cmd = ["python", str(snap_dir / "code" / "experiments/benchmarking/training.py")] + unknown_args

    print(cmd)
    print(" ".join(cmd))

    if not args.dry:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(snap_dir / "code")
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == "__main__":
    main()
