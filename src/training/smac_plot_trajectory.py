import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path
from collections import OrderedDict
import numpy as np
import warnings
from pandas.plotting import parallel_coordinates
from typing import Optional, List
from matplotlib.path import Path
import matplotlib.patches as patches


def fill_trajectory(performance_list, time_list, seed_list: Optional[List[int]] = None, replace_nan=np.NaN):
    """
    https://github.com/automl/plotting_scripts/blob/master/plottingscripts/utils/merge_test_performance_different_times.py
    """
    if len(performance_list) < 2:
        return np.array(performance_list), np.array(time_list).flatten()

    frame_dict = OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    merged = pd.DataFrame(frame_dict)
    merged = merged.ffill()

    performance = merged.values
    time_ = merged.index.values

    performance[np.isnan(performance)] = replace_nan
    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) replace_nan is not set and there are non valid "
                         "numbers in the list\n"
                         "\t(d) any other reason.")

    melted = pd.melt(merged, ignore_index=False, var_name="list_index", value_name="performance")
    melted['seed'] = [np.nan] * len(melted)
    melted['list_index'] = melted['list_index'].apply(int)
    seeds = melted['seed'].to_numpy()
    for i, seed in enumerate(seed_list):
        # print(i)
        # where = np.where(list_index == i)
        seeds[melted['list_index'] == i] = seed
        # print(melted['seed'][melted['list_index'] == i])
    melted['seed'] = seeds

    melted.index.name = "time"

    melted.reset_index(inplace=True)

    return performance, time_, melted


def plot_smac_trajectory(data, key_time, key_performance, key_group):
    groups = data.groupby(key_group)
    performance_list = []
    time_list = []
    seed_list = []
    for group_value, group_df in groups:
        if len(group_df) > 1:
            performance_list.append(group_df[key_performance].to_numpy())
            times = group_df[key_time].to_numpy()
            times[0] = 0
            time_list.append(times)
            seed_list.append(group_df['seed'].unique()[0])  # just one seed per group
        else:
            warnings.warn(f"Short trajectory (length {len(group_df)}) found for trial {group_value}. Do not add to data.")

    performances, times, data_filled = fill_trajectory(
        performance_list=performance_list, time_list=time_list, seed_list=seed_list)

    data_filled.index.name = key_time
    data_filled.rename(columns={'performance': key_performance}, inplace=True)
    data_filled.reset_index(level=0, inplace=True)

    del_ids = data['evaluations'] == 0
    data.drop(data.index[del_ids], inplace=True)

    plot_data = data_filled
    time_unit = "s" if not convert_sec_to_hours else "h"
    new_time_key = f"{key_time} [{time_unit}]"
    plot_data.rename(columns={key_time: new_time_key}, inplace=True)
    key_time = new_time_key
    if convert_sec_to_hours:
        plot_data[key_time] /= 3600


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=plot_data, x=key_time, y=key_performance, ax=ax, marker='o')
    ax.set_yscale('log')
    ax.set_title("SMAC Trajectory")
    fig.set_tight_layout(True)
    plt.show()


def gather_smac_data(outdir, key_group="exp_source"):
    filenames = glob.glob(os.path.join(outdir, "**", "traj.json"), recursive=True)

    data_list = []
    for fname in filenames:
        with open(fname, 'r') as file:
            lines = file.readlines()
        data = [json.loads(line) for line in lines]
        data = pd.DataFrame(data)

        info_fn = "trial_setup.json"
        fname = Path(fname)
        info_fname = fname.parent.parent.parent / info_fn
        with open(info_fname, 'r') as file:
            info = json.load(file)

        agent = info.get("agent", None)
        seed = info.get("seed", None)
        data["seed"] = [seed] * len(data)
        data["agent"] = [agent] * len(data)
        data[key_group] = [info_fname] * len(data)
        data["context_feature_args"] = [info["context_feature_args"]] * len(data)

        data_list.append(data)

    data = pd.concat(data_list)

    return data


def plot_parallel_coordinates__(data, key_hps='incumbent', class_column='seed'):
    data.reset_index(drop=True, inplace=True)
    df = pd.json_normalize(data[key_hps])
    hp_names = list(df.columns)
    df = pd.concat([data, df], axis=1)
    # var_name = "hyperparameter"
    # value_name = "hyperparameter_value"
    # id_vars = [c for c in df.columns if c not in hp_names]
    # df = pd.melt(
    #     df, ignore_index=False, id_vars=id_vars, value_vars=hp_names,
    #     var_name=var_name, value_name=value_name)

    keep_cols = [class_column] + hp_names
    plot_data = df[keep_cols]
    # only plot last incumbent
    groups = plot_data.groupby(class_column)
    new_data = []
    for group_id, group_df in groups:
        if len(group_df) > 1:
            smalldf = pd.DataFrame(group_df.iloc[-1]).T
            new_data.append(smalldf)
    plot_data = pd.concat(new_data)
    plot_data = plot_data.sort_values(class_column)
    figsize = (6, 4)
    dpi = 200
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax = parallel_coordinates(frame=plot_data, class_column=class_column, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title=class_column)
    ax.set_title("Incumbents")
    fig.set_tight_layout(True)
    plt.show()


def plot_parallel_coordinates(
        data: pd.DataFrame,
        key_hps: str = 'incumbent',
        class_column: str = 'seed',
        draw_straight: bool = True
):
    data.reset_index(drop=True, inplace=True)
    df = pd.json_normalize(data[key_hps])
    hp_names = list(df.columns)
    df = pd.concat([data, df], axis=1)
    # var_name = "hyperparameter"
    # value_name = "hyperparameter_value"
    # id_vars = [c for c in df.columns if c not in hp_names]
    # df = pd.melt(
    #     df, ignore_index=False, id_vars=id_vars, value_vars=hp_names,
    #     var_name=var_name, value_name=value_name)

    keep_cols = [class_column] + hp_names
    plot_data = df[keep_cols]
    # only plot last incumbent
    groups = plot_data.groupby(class_column)
    new_data = []
    for group_id, group_df in groups:
        if len(group_df) > 1:
            smalldf = pd.DataFrame(group_df.iloc[-1]).T
            new_data.append(smalldf)
    plot_data = pd.concat(new_data)
    plot_data = plot_data.sort_values(class_column)

    fig, host = plt.subplots()

    ynames = [c for c in plot_data.columns if c != class_column]

    df = plot_data[ynames]
    ys = df.values
    N = ys.shape[0]
    category = plot_data[class_column].to_numpy(dtype=np.int)

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title('Parallel Coordinates Plot', fontsize=18)

    colors = plt.cm.tab10.colors
    # legend_handles = [None for _ in iris.target_names]
    for j in range(N):
        if draw_straight:
            # to just draw straight lines between the axes:
            host.plot(range(ys.shape[1]), zs[j, :], c=colors[(category[j] - 1) % len(colors)])
        else:
            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                             np.repeat(zs[j, :], 3)[1:-1]))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors[category[j]])
            host.add_patch(patch)
    # host.legend(legend_handles, iris.target_names,
    #             loc='lower center', bbox_to_anchor=(0.5, -0.18),
    #             ncol=len(iris.target_names), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


def extract_incumbents(data, key_group: str = "exp_source", key_performance: str = "cost"):
    incumbents = []
    groups = data.groupby(key_group)
    for group_key, group_value in groups:
        # get index of incumbent
        cost_min_idx = group_value[key_performance].argmin()
        inc = pd.DataFrame(group_value.iloc[cost_min_idx]).T
        incumbents.append(inc)
    incumbents = pd.concat(incumbents)

    return incumbents


if __name__ == '__main__':
    """
    Assumed folder structure:
    
    outdir / smag_logs / run_runid / traj.json
    outdir / trial_setup.json
    """
    sns.set_style('white')

    outdir = "/home/benjamin/Dokumente/code/tmp/carl/src/results/optimized/classic_control/CARLCartPoleEnv/0.1_contexthidden"
    outdir = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/optimized/classic_control/CARLMountainCarEnv/0.1_contexthidden"

    convert_sec_to_hours = True
    key_time = "wallclock_time"
    key_performance = "cost"
    key_group = "exp_source"

    data = gather_smac_data(outdir=outdir, key_group=key_group)
    incumbents = extract_incumbents(data=data, key_group=key_group, key_performance=key_performance)

    # plot_smac_trajectory(data=data, key_time=key_time, key_performance=key_performance, key_group=key_group)
    plot_parallel_coordinates(data=data, key_hps='incumbent', class_column='seed', draw_straight=True)










