import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from typing import Union


# TODO move this into seperate module
def collect_results(
        path: Union[str, Path],
        progress_fname: str = "progress.csv",
        eval_fname: str = "evaluations.npz",
        yname: str = "ep_rew_mean",
        from_progress: bool = False,
):
    """
    Assumend folder structure:

    logdir / env_name / name of context feature / {agent}_{seed}
    """
    path = Path(path)

    # context_dirs = [Path(x[0]) for x in os.walk(path)]
    context_dirs = [path / Path(p) for p in os.listdir(path)]
    context_dirs = [p for p in context_dirs if os.path.isdir(p)]
    cf_names = [p.stem for p in context_dirs]

    ids = np.argsort(cf_names)
    context_dirs = np.array(context_dirs)[ids]
    cf_names = np.array(cf_names)[ids]

    # where_baseline = np.where(cf_names == "None")[0]
    # if where_baseline:
    #     idx = where_baseline[0]

    dirs_per_cf = {}
    for i, cf_name in enumerate(cf_names):
        cf_dir = context_dirs[i]
        agent_seed_dirs = os.listdir(cf_dir)
        agent_seed_dirs = [os.path.join(cf_dir, p) for p in agent_seed_dirs]
        agent_seed_dirs = [p for p in agent_seed_dirs if os.path.isdir(p)]
        dirs_per_cf[cf_name] = agent_seed_dirs

    data = {}
    for cf_name, cf_dirs in dirs_per_cf.items():
        D = []
        for cf_dir in cf_dirs:
            cf_dir = Path(cf_dir)
            folder = cf_dir.stem
            if folder == "evaluations":
                continue
            agent, seed = folder.split("_")
            seed = int(seed)

            if from_progress:
                progress_fn = cf_dir / progress_fname
                df = pd.read_csv(progress_fn)
                mean_reward_key = 'rollout/ep_rew_mean'
                time_key = 'time/total_timesteps'
                iteration_key = 'time/iterations'
                if time_key not in df:
                    time_key = 'time/total timesteps'
                if mean_reward_key not in df or time_key not in df:
                    mean_reward_key = 'eval/mean_reward'
                if iteration_key not in df:
                    iteration_key = 'time/episodes'
                n = len(df[time_key])
                D.append(pd.DataFrame({
                    "seed": [seed] * n,
                    "step": df[time_key].to_numpy(),
                    iteration_key: df[iteration_key].to_numpy(),
                    yname: df[mean_reward_key].to_numpy(),
                }))
            else:
                eval_fn = cf_dir / eval_fname
                try:
                    eval_data = np.load(str(eval_fn))
                    timesteps = eval_data["timesteps"]
                    ep_lengths = eval_data["ep_lengths"]
                    mean_ep_length = np.mean(ep_lengths, axis=1)
                    iteration = None
                    y = np.mean(eval_data["results"], axis=1)
                    n = len(timesteps)
                    D.append(pd.DataFrame({
                        "seed": [seed] * n,
                        "step": timesteps,
                        "iteration": [iteration] * n,
                        yname: y,
                        "mean_ep_length": mean_ep_length
                    }))
                except Exception as e:
                    print(e)

        if D:
            D = pd.concat(D)
            data[cf_name] = D

    # metadata = {
    #     "env_name": env_name,
    # }
    return data


if __name__ == "__main__":
    paths = [
        # "results/base_vs_context/classic_control/MetaPendulumEnv/0.5_contexthidden",
        # "results/base_vs_context/classic_control/CARLPendulumEnv/0.5_changingcontextvisible",

        # bipedal
        "results/base_vs_context/box2d/CARLBipedalWalkerEnv/0.1_contexthidden",
        "results/base_vs_context/box2d/CARLBipedalWalkerEnv/0.25_contexthidden",
        "results/base_vs_context/box2d/CARLBipedalWalkerEnv/0.5_contexthidden"
    ]

    fname_id = "" # "_comparevisibility"

    from_progress = False
    plot_comparison = True
    plot_mean_performance = False
    plot_ep_lengths = True
    paperversion = True
    libname = "CARL"
    hue = None #"seed"
    plot_combined = True
    if len(paths) == 1:
        plot_combined = False

    sns.set_context("paper")
    labelfontsize = None
    titlefontsize = None
    ticklabelsize = None
    legendfontsize = None
    if paperversion:
        sns.set_context("paper")
        labelfontsize = 12
        titlefontsize = 12
        ticklabelsize = 10
        legendfontsize = 10
    sns.set_style("whitegrid")
    fig = None
    if plot_combined:
        figsize = (8, 6) if not paperversion else (6, 3)
        dpi = 200
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ncols = len(paths)
        axes = fig.subplots(nrows=1, ncols=ncols, sharey=True)
        fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, bottom=0.01, top=0.99)
    for idx_plot, path in enumerate(paths):
        # if idx_plot != 1:
        #     continue
        path = Path(path)
        print(path)


        xname = "step"
        yname = "ep_rew_mean"
        default_name = "None"  # identifier for environment with standard context
        savepath = path
        if "singlecontext" in str(path):
            std = path.parts[-3].split("_")[1]
            envtype = path.parts[-2]
            envname = path.parts[-1]
        else:
            std = path.parts[-1].split("_")[0]
            envtype = path.parts[-3]
            envname = path.parts[-2]
        hc = "cv"
        if np.any(["hidecontext" in p for p in path.parts]) or np.any(["contexthidden" in p for p in path.parts]):
            hc = "ch"

        contextchanging_str = ""
        if np.any(["changing" in p for p in path.parts]):
            contextchanging_str = "changing "

        if "eimer" in path.parts:
            default_name = "vanilla"
            parts = path.parts
            std = parts[-2].split("_")[-1]
            envtype = parts[-3]
            envname = parts[-1]
            savepath = Path(f"results/singlecontextfeature_{std}/{envtype}/{envname}")
            savepath.mkdir(parents=True, exist_ok=True)

        fromprogressstr = "" if not from_progress else "_fromprogress"
        fname_comparison = savepath / f"{envname}_{std}_{hc}_ep_rew_mean_over_{xname}{fromprogressstr}.png"
        fname_comparison2 = Path(f"results/base_vs_context/{envtype}/{envname}")
        fname_comparison2.mkdir(parents=True, exist_ok=True)
        fname_comparison2 /= f"reward_over_time_{std}_{hc}.png"
        fname_reward = savepath / f"{envname}_ep_rew_mean_mean_std.png"

        data = collect_results(path, yname=yname, from_progress=from_progress)
        env_name = envname
        if "Meta" in env_name:
            env_name = env_name.replace("Meta", libname)

        ylims = None
        xlims = (5e3, 1e6)
        if "Bipedal" in env_name:
            ylims = (-1000, 500)
            # ylims = None
        elif "Pendulum" in env_name:
            if paperversion:
                ylims = (-1500, -100)
            # ylims = None

        if "hidden" in str(path) or "hide" in str(path):
            contextvisiblity_str = "hidden"
        else:
            contextvisiblity_str = "visible"

        color_palette_name = "colorblind"
        color_default_context = "black"

        if plot_comparison:
            if fig is None:
                figsize = (8, 6) if not paperversion else (4, 3)
                dpi = 200
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = fig.add_subplot(111)
            else:
                ax = axes[idx_plot]
            delete_keys = [k for k in data.keys() if "ignore" in k]
            for k in delete_keys:
                del data[k]
            n = len(data)
            # n_colors_in_palette = n
            # if color_palette_name in sns.palettes.SEABORN_PALETTES:
            #     n_colors_in_palette = sns.palettes.SEABORN_PALETTES[color_palette_name]
            colors = sns.color_palette(color_palette_name, n)
            legend_handles = []
            labels = []
            for i, (cf_name, df) in enumerate(data.items()):
                color = colors[i]
                if cf_name == default_name:
                    color = color_default_context
                print(df[yname].min(), df[yname].max())
                ax = sns.lineplot(data=df, x=xname, y=yname, ax=ax, color=color, marker='', hue=hue)
                legend_handles.append(Line2D([0], [0], color=color))
                labels.append(cf_name)
            if ylims:
                ax.set_ylim(*ylims)
            ax.set_xlim(*xlims)
            ax.set_xscale("log")
            # ax.set_yscale("log")
            title = f"{env_name}, $\sigma_{{rel}}={std}$, context {contextvisiblity_str} \n{str(path)}"
            if paperversion or True:
                title = f"{env_name}\n$\sigma_{{rel}}={std}$, {contextchanging_str}context {contextvisiblity_str}"
                title = f"$\sigma_{{rel}}={std}$"
            ax.set_title(title, fontsize=titlefontsize)
            ax.set_xlabel("step", fontsize=labelfontsize)
            ax.set_ylabel("mean reward\nacross instances $\mathcal{I}_{train}$", fontsize=labelfontsize)
            ax.tick_params(labelsize=ticklabelsize)

            # Sort labels, put default name at front
            if default_name in labels:
                idx = labels.index(default_name)
                name_item = labels.pop(idx)
                labels.insert(0, name_item)
                handle_item = legend_handles.pop(idx)
                legend_handles.insert(0, handle_item)

            if (idx_plot == 1 and contextvisiblity_str == "hidden") or not plot_combined:
                ncols = len(legend_handles)
                legend = fig.legend(
                    handles=legend_handles,
                    labels=labels,
                    loc='lower center',
                    title="varying context feature",
                    ncol=ncols,
                    fontsize=legendfontsize,
                    columnspacing=0.5,
                    handletextpad=0.5,
                    handlelength=1.5,
                    bbox_to_anchor=(0.5, 0.205)
                )
            fig.set_tight_layout(True)

            if not plot_combined:
                plt.show()
                fig.savefig(fname_comparison, bbox_inches="tight")
                if not from_progress:
                    fig.savefig(fname_comparison2, bbox_inches="tight")
    if plot_combined:
        fig.set_tight_layout(True)
        plt.show()
        fname = os.path.join(os.path.commonpath(paths), f"{env_name}_mean_ep_rew_over_step_{contextvisiblity_str}{fname_id}.png")
        fig.savefig(fname, bbox_inches="tight")

        # if plot_ep_lengths and "mean_ep_length" in data:
        #     pass

        # if plot_mean_performance:
        #     sns.set_style("darkgrid")
        #     means = []
        #     for cf_name, df in data.items():
        #         mean = df[yname].mean()
        #         std = df[yname].std()
        #         means.append({"context_feature": cf_name, "mean": mean, "std": std})
        #     means = pd.DataFrame(means)
        #
        #     n = len(means)
        #     colors = np.array(sns.color_palette(color_palette_name, n))
        #     idx = means["context_feature"] == default_name
        #     colors[idx] = mpl.colors.to_rgb(color_default_context)
        #
        #     figsize = (8, 6)
        #     dpi = 200
        #     fig = plt.figure(figsize=figsize, dpi=200)
        #     axes = fig.subplots(nrows=2, ncols=1, sharex=True)
        #
        #     ax = axes[0]
        #     ax = sns.barplot(data=means, x="context_feature", y="mean", ax=ax, palette=colors)
        #     ax.set_xlabel("")
        #     ax = axes[1]
        #     ax = sns.barplot(data=means, x="context_feature", y="std", ax=ax, palette=colors)
        #     xticklabels = means["context_feature"]
        #     ax.set_xticklabels(xticklabels, rotation=30, fontsize=9, ha="right")
        #     title = f"{env_name}\n{str(path)}"
        #     fig.suptitle(title)
        #     fig.set_tight_layout(True)
        #     fig.savefig(fname_reward, bbox_inches="tight")
        #     plt.show()




