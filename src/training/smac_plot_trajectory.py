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


def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
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
    melted.index.name = "time"

    return performance, time_, melted


def plot_smac_trajectory(data, key_time, key_performance, key_group):
    groups = data.groupby(key_group)
    performance_list = []
    time_list = []
    for group_value, group_df in groups:
        if len(group_df) > 1:
            performance_list.append(group_df[key_performance].to_numpy())
            times = group_df[key_time].to_numpy()
            times[0] = 0
            time_list.append(times)
        else:
            warnings.warn(f"Short trajectory (length {len(group_df)}) found for trial {group_value}. Do not add to data.")

    performances, times, data_filled = fill_trajectory(performance_list=performance_list, time_list=time_list)

    data_filled.index.name = key_time
    data_filled.rename(columns={'list_index': 'seed', 'performance': key_performance}, inplace=True)
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

        data_list.append(data)

    data = pd.concat(data_list)

    return data


def plot_parallel_coordinates(data, key_hps='incumbent', class_column='seed'):
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
    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    """
    Assumed folder structure:
    
    outdir / smag_logs / run_runid / traj.json
    outdir / trial_setup.json
    """
    outdir = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/optimized/classic_control/CARLPendulumEnv/0.75_contexthidden/g/"

    convert_sec_to_hours = True
    key_time = "wallclock_time"
    key_performance = "cost"
    key_group = "exp_source"

    data = gather_smac_data(outdir=outdir, key_group=key_group)
    # plot_smac_trajectory(data=data, key_time=key_time, key_performance=key_performance, key_group=key_group)
    plot_parallel_coordinates(data=data, key_hps="incumbent", class_column='seed')





