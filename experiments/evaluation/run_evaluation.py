import time
import os
import sys
import numpy as np
import argparse
import pathlib, shutil
from datetime import datetime
from subprocess import Popen, DEVNULL
from rich import print
import glob
import regex as re


def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        help="Path of submitit folder. If empty, uses single cfg.results_path.",
    )
    args, unknown_args = parser.parse_known_args()  # unknown args are hydra commands
    # unknown_args = [f"'{a}'" for a in unknown_args]

    add_multirun_flag = False
    if unknown_args and unknown_args[-1] == "-m":
        unknown_args.pop(-1)
        add_multirun_flag = True

    # Find subfolders with multiruns
    if args.result_dir is not None:
        filenames = list(glob_re(r"\d\d*", os.listdir(args.result_dir)))
        filenames.sort(key=float)
        result_paths = [f"'{os.path.join(args.result_dir, f)}'" for f in filenames]
        result_paths_str = ",".join(result_paths)
        result_paths_arg = f"results_path={result_paths_str}"
        unknown_args.append(result_paths_arg)

    if add_multirun_flag:
        unknown_args += ["-m"]

    cmd = ["python", "experiments/evaluation/evaluate.py"] + unknown_args

    print(" ".join(cmd))

    if not args.dry:
        env = os.environ.copy()
        # env["PYTHONPATH"] = str(snap_dir / "code")
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == "__main__":
    main()
