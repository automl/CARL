import os
from pathlib import Path
from typing import Union, Dict, Optional
import glob
import json

import numpy as np
import pandas as pd


def collect_results(
        path: Union[str, Path],
        progress_fname: str = "progress.csv",
        eval_fname: str = "evaluations.npz",
        yname: str = "ep_rew_mean",
        from_progress: bool = False,
        dirs_per_cf: Dict = None,
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

    if dirs_per_cf is None:
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


def load_trial_setup(path: Union[str, Path], info_fn: str = "trial_setup.json"):
    if info_fn.endswith(".ini"):
        raise NotImplementedError
    path = Path(path)
    info_fname = path / info_fn
    with open(info_fname, 'r') as file:
        info = json.load(file)
    return info, info_fname


def extract_info(path: Union[str, Path], info_fn: str = "trial_setup.json"):
    info, info_fname = load_trial_setup(path=path, info_fn=info_fn)

    agent = info.get("agent", None)
    seed = info.get("seed", None)
    evaluation_protocol_mode = info.get("evaluation_protocol_mode", None)
    data = {
        "env": info["env"],
        "seed": seed,
        "agent": agent,
        "config_filename": info_fname,
        "context_feature_args": info["context_feature_args"],
        "context_variation_magnitude": info["default_sample_std_percentage"],
        "context_visible": not info["hide_context"]
    }
    if evaluation_protocol_mode is not None:
        data["evaluation_protocol_mode"] = evaluation_protocol_mode

    return data


def gather_results(
        path: Union[str, Path],
        progress_fname: str = "progress.csv",
        eval_fname: str = "evaluations.npz",
        yname: str = "ep_rew_mean",
        from_progress: bool = False,
        dirs_per_cf: Dict = None,
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

        D = None
        if from_progress:
            progress_fn = result_dir / progress_fname
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

            step = df[time_key].to_numpy()
            iter_key = "iteration" if "iteration" in iteration_key else "episode"

            D = pd.DataFrame({
                "step": df[time_key].to_numpy(),
                iter_key: df[iteration_key].to_numpy(),
                yname: df[mean_reward_key].to_numpy(),
            })
        else:
            eval_fn = result_dir / eval_fname
            if not eval_fn.is_file():
                print(f"Eval fn {eval_fn} does not exist. Skip.")
                continue

            eval_data = np.load(str(eval_fn))
            timesteps = eval_data["timesteps"]
            episode_lengths = eval_data["ep_lengths"]
            instance_ids = np.squeeze(eval_data["episode_instances"])
            episode_rewards = eval_data["results"]

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
            })

        # merg info and results
        n = len(D)
        for k, v in info.items():
            D[k] = [v] * n

        if D is not None:
            data.append(D)
    
    if data:
        data = pd.concat(data)

    return data


if __name__ == '__main__':
    path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/base_vs_context/classic_control"
    results = gather_results(path=path)
