import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from typing import Union, Dict
from pathlib import Path
import os
import glob
import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

from src.eval.gather_data import extract_info
from src.experiments.train_on_protocol import get_context_features


def gather_results(
        path: Union[str, Path],
        eval_file_id: Union[str, Path] = "eval/evaluation_protocol/*.npz",
        trial_setup_fn: str = "trial_setup.json"
):
    dirs = glob.glob(os.path.join(path, "**", trial_setup_fn), recursive=True)

    data = []
    for result_dir in dirs:
        result_dir = Path(result_dir).parent
        # agent
        # seed
        # context feature args
        info = extract_info(path=result_dir, info_fn=trial_setup_fn)
        mode = info["evaluation_protocol_mode"]
        eval_fns = glob.glob(str(result_dir / eval_file_id), recursive=True)
        for eval_fn in eval_fns:
            eval_fn = Path(eval_fn)
            D = None
            if not eval_fn.is_file():
                print(f"Eval fn {eval_fn} does not exist. Skip.")
                continue

            eval_data = np.load(str(eval_fn))
            timesteps = eval_data["timesteps"]
            episode_lengths = eval_data["ep_lengths"]
            instance_ids = np.squeeze(eval_data["episode_instances"])
            episode_rewards = eval_data["results"]

            key = None
            if "test_id" in eval_data:
                key = "test_id"
            elif "context_distribution_type" in eval_data:
                key = "context_distribution_type"

            context_distribution_type = eval_data[key] if key is not None else None

            timesteps = np.concatenate([timesteps[:, np.newaxis]] * episode_rewards.shape[-1], axis=1)

            timesteps = timesteps.flatten()
            instance_ids = instance_ids.flatten()
            instance_ids = instance_ids.astype(dtype=np.int)
            episode_rewards = episode_rewards.flatten()
            episode_lengths = episode_lengths.flatten()

            n = len(timesteps)

            D = pd.DataFrame({
                "step": timesteps,
                "instance_id": instance_ids,
                "episode_reward": episode_rewards,
                "episode_length": episode_lengths,
                "context_distribution_type": context_distribution_type,
                "mode": mode
            })

            # merge info and results
            n = len(D)
            for k, v in info.items():
                D[k] = [v] * n

            if D is not None:
                data.append(D)

    if data:
        data = pd.concat(data)

    return data


if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/evaluation_protocol/base_vs_context/classic_control/CARLCartPoleEnv"
    # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/tmp/test_logs/CARLCartPoleEnv"
    results = gather_results(path=path)

    assert results["env"].nunique() == 1
    env = results["env"].iloc[0]

    context_features = get_context_features(env_name=env)

    modes = results["mode"].unique()
    n_protocols = len(modes)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    axes = fig.subplots(nrows=1, ncols=n_protocols, sharex=True, sharey=True)

    groups = results.groupby("mode")
    cf0, cf1 = context_features
    xlim = (cf0.lower, cf0.upper)
    ylim = (cf1.lower, cf1.upper)

    for i, (group_id, group_df) in enumerate(groups):
        ax = axes[i]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        mode = modes[i]

        context_distribution_types = results["context_distribution_type"].unique()

        performances = {}
        for context_distribution_type in context_distribution_types:
            sub_df = group_df[group_df["context_distribution_type"] == context_distribution_type]
            performance = np.mean(sub_df["episode_reward"])
            performances[context_distribution_type] = performance


        # Draw Quadrants
        patches = []

        colors = sns.color_palette("colorblind")
        color_T = colors[0]
        color_I = colors[1]
        color_ES = colors[2]
        color_EB = colors[3]
        color_IC = colors[4]

        ec_test = "black"
        markerfacecolor_alpha = 0.

        patch_kwargs = dict(zorder=0, linewidth=0., )

        # Extrapolation along single factors, Q_ES
        xy = (cf0.mid, cf1.lower)
        width = cf0.upper - cf0.mid
        height = cf1.mid - cf1.lower
        Q_ES = Rectangle(xy=xy, width=width, height=height, color=color_ES, **patch_kwargs)
        patches.append(Q_ES)

        xy = (cf0.lower, cf1.mid)
        height = cf1.upper - cf1.mid
        width = cf0.mid - cf0.lower
        Q_ES = Rectangle(xy=xy, width=width, height=height, color=color_ES, **patch_kwargs)
        patches.append(Q_ES)

        # Extrapolation along both factors
        xy = (cf0.mid, cf1.mid)
        height = cf1.upper - cf1.mid
        width = cf0.upper - cf0.mid
        Q_EB = Rectangle(xy=xy, width=width, height=height, color=color_EB, **patch_kwargs)
        patches.append(Q_EB)

        # Interpolation
        if mode == "A":
            xy = (cf0.lower, cf1.lower)
            height = cf1.mid - cf1.lower
            width = cf0.mid - cf0.lower
            Q_I = Rectangle(xy=xy, width=width, height=height, color=color_I, **patch_kwargs)
            patches.append(Q_I)
        elif mode == "B":
            xy = (cf0.lower, cf1.lower)
            width = cf0.mid - cf0.lower
            height = cf1.lower_constraint - cf1.lower
            Q_I = Rectangle(xy=xy, width=width, height=height, color=color_I, **patch_kwargs)
            patches.append(Q_I)

            width = cf0.lower_constraint - cf0.lower
            height = cf1.mid - cf1.lower
            Q_I = Rectangle(xy=xy, width=width, height=height, color=color_I, **patch_kwargs)
            patches.append(Q_I)

        # Combinatorial Interpolation
        if mode == "B":
            xy = (cf0.lower_constraint, cf1.lower_constraint)
            height = cf1.mid - cf1.lower_constraint
            width = cf0.mid - cf0.lower_constraint
            Q_IC = Rectangle(xy=xy, width=width, height=height, color=color_IC, **patch_kwargs)
            patches.append(Q_IC)
        elif mode == "C":
            xy = (cf0.lower, cf1.lower)
            height = cf1.mid - cf1.lower
            width = cf0.mid - cf0.lower
            Q_IC = Rectangle(xy=xy, width=width, height=height, color=color_IC, **patch_kwargs)
            patches.append(Q_IC)

        for patch in patches:
            ax.add_patch(patch)

        ax.set_title(mode)

    fig.set_tight_layout(True)
    plt.show()

