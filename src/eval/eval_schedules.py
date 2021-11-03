import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_hps(policy_file):
    raw_policy = []
    with open(policy_file, "rt") as fp:
        for row in fp.readlines():
            parsed_row = json.loads(row)
            raw_policy.append(tuple(parsed_row))

    policy = []
    last_new_tag = None
    last_old_conf = None
    for (old_tag, new_tag, old_step, new_step, old_conf, new_conf) in reversed(raw_policy):
        if last_new_tag and old_tag != last_new_tag:
            break
        last_new_tag = new_tag
        last_old_conf = old_conf
        policy.append((new_step, new_conf))

    # return last_old_conf, list(reversed(policy))
    return list(reversed(policy))


def gather_data(path, visibility: str):
    paths = glob.glob(os.path.join(path, "*"))
    pattern = "pbt_policy_[0-9A-Fa-f]{5}_[0-9]{5}.txt$"
    schedule_fnames = []
    for p in paths:
        match = re.search(pattern, p)
        if match is not None:
            schedule_fnames.append(p)
    schedule_fnames.sort()
    all_schedules = [load_hps(p) for p in schedule_fnames]
    data = []
    for i, schedules in enumerate(all_schedules):
        configs = []
        identifier = Path(schedule_fnames[i]).stem
        for step, config in schedules:
            config["step"] = step
            configs.append(config)
        df = pd.DataFrame(configs)
        df["id"] = [identifier] * len(df)
        data.append(df)
    data = pd.concat(data)
    data["visibility"] = [visibility] * len(data)

    return data


if __name__ == "__main__":
    # Acrobot
    path_hidden = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLAcrobotEnv/ray/pb2_mountaincar_gravity_hidden/"
    path_visible = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLAcrobotEnv/ray/pb2_mountaincar_gravity/"

    # LunarLander
    # path_hidden = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLLunarLanderEnv/ray/pb2_ll_gravity_hidden/"
    # path_visible = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLLunarLanderEnv/ray/pb2_ll_gravity/"

    # Pendulum
    # path_hidden = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLPendulumEnv/ray/pb2_pendulum_hidden"
    # path_visible = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLPendulumEnv/ray/pb2_pendulum"

    target_dir = "/home/benjamin/Dokumente/code/tmp/CARL/src/results"
    actual_steps_per_step = 1  # 4096
    data_hidden = gather_data(path_hidden, "hidden")
    data_visible = gather_data(path_visible, "visible")
    data_hidden["step"] *= actual_steps_per_step
    data_visible["step"] *= actual_steps_per_step
    data_hidden.rename(columns={"step": "PB2_iteration"}, inplace=True)
    data_visible.rename(columns={"step": "PB2_iteration"}, inplace=True)
    data = pd.concat((data_visible, data_hidden))

    cols = [c for c in data.columns if c not in ["id", "visibility", "step", "PB2_iteration"]]

    hue = "id"
    xname = "PB2_iteration"

    fig = plt.figure(figsize=(4, 8))
    axes = fig.subplots(nrows=len(cols), ncols=2, sharey=False, sharex=True)
    for i, (ax_h, ax_v) in enumerate(axes):
        yname = cols[i]
        ax_h = sns.lineplot(data=data_hidden, x=xname, y=yname, hue=hue, ax=ax_h)
        ax_h.get_legend().remove()

        ax_v = sns.lineplot(data=data_visible, x=xname, y=yname, hue=hue, ax=ax_v)
        ax_v.get_legend().remove()
        ax_v.set_ylabel(None)
        # ax.set_title()

        ylim_h = ax_h.get_ylim()
        ylim_v = ax_v.get_ylim()
        ylim = (min(ylim_h[0], ylim_v[0]), max(ylim_h[1], ylim_v[1]))
        ax_h.set_ylim(*ylim)
        ax_v.set_ylim(*ylim)

        if i == 0:
            ax_h.set_title("hidden")
            ax_v.set_title("visible")
    fig.set_tight_layout(True)
    plt.show()

    path = Path(path_visible)
    env_idx = [part.startswith("CARL") for part in path.parts]
    env_name = np.array(path.parts)[env_idx][0]
    fig_fname = os.path.join(target_dir, f"pb2_hp_schedule_{env_name}.png")
    fig.savefig(fig_fname, dpi=200, bbox_inches="tight")





