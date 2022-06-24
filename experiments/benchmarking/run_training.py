import time
import os
import sys
import argparse
import pathlib, shutil
from datetime import datetime
from subprocess import Popen, DEVNULL
from rich import print


def make_code_snap(experiment, slurm_dir='exp_sweep'):
    now = datetime.now()
    snap_dir = pathlib.Path.cwd() / slurm_dir
    snap_dir /= now.strftime('%Y.%m.%d')
    snap_dir /= now.strftime('%H%M%S') + f'_{experiment}'
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
        'experiments/carlbench'
    ]
    src_dir = pathlib.Path.cwd()
    for dir in dirs_to_copy:
        copy_dir(dir)

    # Copy repository / environment infos
    rootfiles = src_dir.glob("*")
    for f in rootfiles:
        if f.is_file() and not str(f.name).startswith("."):
            shutil.copyfile(f, snap_dir / 'code' / f.name)

    return snap_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true')
    args, unknown_args = parser.parse_known_args()  # unknown args are hydra commands
    unknown_args = [f"'{a}'" for a in unknown_args]

    experiment = 'benchmark'
    snap_dir = make_code_snap(experiment)
    print("Snap dir:", str(snap_dir))

    cmd = ['python', str(snap_dir / 'code' / 'training.py')] + unknown_args

    print(" ".join(cmd))

    if not args.dry:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
