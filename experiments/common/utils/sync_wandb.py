"""
Sync all wandb (offline) runs from an experiment sweep folder (created by hydra).
Add your exp path to the command line and the script will find every folder
named "latest-run".

"""
import os
import sys
from pathlib import Path
from subprocess import DEVNULL, Popen

from tqdm import tqdm

if __name__ == "__main__":
    outdir = sys.argv[1] or "./exp_sweep/2022-09-21/12-14-29"
    path = Path(outdir)
    wandb_dirs = list(path.rglob("*latest-run"))
    wandb_dirs.sort()
    print(len(wandb_dirs))
    for wandb_dir in tqdm(wandb_dirs, total=len(wandb_dirs)):
        env = os.environ.copy()
        abs_path = wandb_dir.resolve()
        cmd = f"source /opt/conda/bin/activate dacbo; wandb sync {abs_path}"
        p = Popen(cmd, shell=True)
        p.communicate()
