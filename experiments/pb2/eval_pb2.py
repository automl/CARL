import glob
import os
from pathlib import Path
from typing import List, Dict
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.common.eval.gather_data import collect_results


def clean_worker_dirs(dirs: List[str]):
    pattern = "/p[0-9]+$"
    clean_dirs = []
    for d in dirs:
        match = re.search(pattern, d)
        if match is not None:
            clean_dirs.append(d)
    return clean_dirs


def get_worker_ids(dirs: List[str]) -> Dict[str, int]:
    pattern = "/p[0-9]+$"
    worker_ids = {}
    for d in dirs:
        match = re.search(pattern, d)
        if match is not None:
            group = match.group()
            worker_id = int(group[2:])
            worker_ids[d] = worker_id
    return worker_ids


def clean_stddirs(dirs: List[str]) -> Dict[str, int]:
    pattern = r"/std_\d+\.\d+$"
    std_dirs = {}
    for d in dirs:
        match = re.search(pattern, d)
        if match is not None:
            group = match.group()
            std = float(group[5:])
            std_dirs[d] = std
    return std_dirs


def get_data_0(path, env_name, p_id_visible, p_id_hidden, from_progress):
    """
    assumend folder structure
    # carl/results/<env_family>/<id_hiddenvisible>/<worker_dir>/std_<std>/<env_name>/<context_feature_name>/<agent_seed>
    """
    worker_dirs_visible = glob.glob(str(path / p_id_visible / "*"))
    worker_dirs_visible = clean_worker_dirs(worker_dirs_visible)
    worker_ids_visible = get_worker_ids(worker_dirs_visible)

    worker_dirs_hidden = glob.glob(str(path / p_id_hidden / "*"))
    worker_dirs_hidden = clean_worker_dirs(worker_dirs_hidden)
    worker_ids_hidden = get_worker_ids(worker_dirs_hidden)

    visibilities = ["visible", "hidden"]
    worker_ids_list = [worker_ids_visible, worker_ids_hidden]
    data = []
    for visibility, worker_ids in zip(visibilities, worker_ids_list):
        for wdir, worker_id in worker_ids.items():
            std_dirs = glob.glob(str(Path(wdir) / "*"))
            std_dirs = clean_stddirs(std_dirs)

            for std_dir, std in std_dirs.items():
                D = collect_results(Path(std_dir) / env_name, from_progress=from_progress)
                for cf, df in D.items():
                    df["worker_id"] = [worker_id] * len(df)
                    df["std"] = [std] * len(df)
                    df["context_feature"] = [cf] * len(df)
                    df["visibility"] = [visibility] * len(df)
                    data.append(df)
    data = pd.concat(data)
    return data


def get_data_1(path, env_name, p_id_visible, p_id_hidden, from_progress, std=0.1, no_cf_dir=False, cf_name=""):
    """
    assumend folder structure
    # carl/results/<env_family>/<id_hiddenvisible>/<env_name>/<worker_dir>/<context_feature_name>/<agent_seed>
    """
    worker_dirs_visible = glob.glob(str(path / p_id_visible / env_name / "*"))
    worker_dirs_visible = clean_worker_dirs(worker_dirs_visible)
    worker_ids_visible = get_worker_ids(worker_dirs_visible)

    worker_dirs_hidden = glob.glob(str(path / p_id_hidden / env_name / "*"))
    worker_dirs_hidden = clean_worker_dirs(worker_dirs_hidden)
    worker_ids_hidden = get_worker_ids(worker_dirs_hidden)

    visibilities = ["visible", "hidden"]
    worker_ids_list = [worker_ids_visible, worker_ids_hidden]
    data = []
    for visibility, worker_ids in zip(visibilities, worker_ids_list):
        for wdir, worker_id in worker_ids.items():
            dirs_per_cf = None
            if not no_cf_dir:
                dirs_per_cf = {cf_name: glob.glob(os.path.join(wdir, "*"))}
            D = collect_results(wdir, from_progress=from_progress, dirs_per_cf=dirs_per_cf)
            for cf, df in D.items():
                df["worker_id"] = [worker_id] * len(df)
                df["std"] = [std] * len(df)
                df["context_feature"] = [cf] * len(df)
                df["visibility"] = [visibility] * len(df)
                data.append(df)
    data = pd.concat(data)
    return data


if __name__ == "__main__":
    # assumed folder structure
    # src/results/<env_family>/<id_hiddenvisible>/<worker_dir>/std_<std>/<env_name>/<context_feature_name>/<agent_seed>
    structure_0 = False
    outdir = "/home/eimer/Dokumente/git/meta-gym/src/results/classic_control/"
    outdir = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/classic_control/"
    path = Path(outdir)
    env_name = "CARLLunarLanderEnv"
    env_name = "CARLAcrobotEnv"
    context_feature_name = "link_length_1"
    p_id_visible = "pbt_hps"
    p_id_hidden = "pbt_hps_hidden"
    from_progress = True



    if structure_0:
        data = get_data_0(path, env_name, p_id_visible, from_progress, p_id_hidden)
    else:
        data = get_data_1(path, env_name, p_id_visible, p_id_hidden, from_progress, no_cf_dir=True, cf_name=context_feature_name)


    # data contains:
    # ['seed', 'step', 'iteration', 'ep_rew_mean', 'mean_ep_length', 'worker_id', 'std', 'context_feature', 'visibility']

    # plot single workers
    figsize = (5, 3)
    dpi = 200
    palette_name = "colorblind"
    unique_worker_ids = data["worker_id"].unique()
    palette = dict(zip(unique_worker_ids, sns.color_palette(palette=palette_name, n_colors=len(unique_worker_ids))))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)

    ax_h = axes[0]  # axis for hidden context
    ax_v = axes[1]  # axis for visible context

    data_h = data[data["visibility"] == "hidden"]
    data_v = data[data["visibility"] == "visible"]

    ax_h = sns.lineplot(data=data_h, x="step", y="ep_rew_mean", hue="worker_id", style="seed", ax=ax_h, palette=palette)
    ax_v = sns.lineplot(data=data_v, x="step", y="ep_rew_mean", hue="worker_id", style="seed", ax=ax_v, palette=palette)

    ax_h.set_title("hidden")
    ax_v.set_title("visible")
    ax_h.set_ylabel("mean reward")
    try:
        ax_h.get_legend().remove()
    except:
        pass
    try:
        ax_v.get_legend().remove()
    except:
        pass

    fig.set_tight_layout(True)
    figfname = Path(os.getcwd()) / "results" / f"pb2workers_{env_name}_hidden_vs_visible.png"
    fig.savefig(figfname, bbox_inches="tight")
    plt.show()
