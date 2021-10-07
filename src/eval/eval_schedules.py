import os
import re
import glob
import json
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
    path = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLAcrobotEnv/ray/pb2_mountaincar_gravity_hidden/"
    path_hidden = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLLunarLanderEnv/ray/pb2_ll_gravity_hidden/"
    path_visible = "/home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLLunarLanderEnv/ray/pb2_ll_gravity/"
    data_hidden = gather_data(path_hidden, "hidden")
    data_visible = gather_data(path_visible, "visible")
    data = pd.concat((data_visible, data_hidden))


    fig = plt.figure(figsize=(8, 6))
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)
    ax = axes[0]
    ax = sns.lineplot(data=data_hidden, x="step", y="learning_rate", hue="id", ax=ax)
    ax.set_title("hidden")

    ax = axes[1]
    ax = sns.lineplot(data=data_visible, x="step", y="learning_rate", hue="id", ax=ax)
    ax.set_title("visible")
    # ax.set_title()
    fig.set_tight_layout(True)
    plt.show()




