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
from typing import List, Union
from pathlib import Path
from functools import partial


def filter_fn(string, pattern):
    allowed = False
    match = re.compile(pattern).match(string)
    if match and ".submitit" not in string:
        allowed = True
    return allowed 


def glob_re(pattern, strings):
    filter_fn_partial = partial(filter_fn, pattern=pattern)
    return filter(filter_fn_partial, strings)


def get_all_subpaths(rdir: str):
    paths = []
    for root, dirs, files in os.walk(rdir):
         for p in dirs:
             p = os.path.join(root, p)
             paths.append(p)
    return paths


def find_multirun_paths(result_dir: Union[str, List[str]]) -> List[str]:
    """
    Find all folders with a number as name

    These folders are the folders from the hydra multirun.

    Parameters
    ----------
    result_dir : Union[str, List[str]]
        Path(s) to hydra result dir containing multirun folders.

    Returns
    -------
    List[str]
        Paths to individual runs

    """
    if type(result_dir) != list:
        result_dir = [result_dir]
    result_paths = []
    for rdir in result_dir:
        dirs = get_all_subpaths(rdir=rdir)
        filenames = list(glob_re(r".*/\d*$", dirs))
        filenames.sort(key=lambda x: float(x.split("/")[-1]))
        result_paths.extend(filenames)

    if len(result_paths) == 0:
        raise ValueError(f"Could not find any result paths in {result_dir}.")

    return result_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        nargs="+",
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
        result_paths = find_multirun_paths(result_dir=args.result_dir)
        result_paths = [Path(p) for p in result_paths]
        result_paths = [f"{p.resolve()}" for p in result_paths]
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
