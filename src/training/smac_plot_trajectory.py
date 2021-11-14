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


if __name__ == '__main__':
    """
    Assumed folder structure:
    
    outdir / smag_logs / run_runid / traj.json
    outdir / trial_setup.json
    """
    fname = "/home/benjamin/Dokumente/code/tmp/CARL/src/training/smac3-output_2021-11-12_10:59:28_979602/run_1593118232/traj.json"
    outdir = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/optimized/classic_control/CARLPendulumEnv/0.75_contexthidden/g/"

    key_time = "wallclock_time"
    key_performance = "cost"

    filenames = glob.glob(os.path.join(outdir, "**", "traj.json"), recursive=True)

    data_list = []
    performance_list = []
    time_list = []
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

        if len(data) > 1:
            performance_list.append(data[key_performance].to_numpy())
            time_data = data[key_time].to_numpy()
            time_data[0] = 0
            time_list.append(time_data)
            data_list.append(data)
        else:
            warnings.warn(f"Short trajectory (length {len(data)}) found for trial {info_fname}. Do not add to data.")

    data = pd.concat(data_list)
    performances, times, data_filled = fill_trajectory(performance_list=performance_list, time_list=time_list)

    data_filled.index.name = key_time
    data_filled.rename(columns={'list_index': 'seed', 'performance': key_performance}, inplace=True)

    del_ids = data['evaluations'] == 0
    data.drop(data.index[del_ids], inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=data_filled, x=key_time, y=key_performance, ax=ax, marker='o')
    ax.set_yscale('log')
    fig.set_tight_layout(True)
    plt.show()

