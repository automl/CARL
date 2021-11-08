import sys
sys.path.append("..")
import matplotlib
matplotlib.use('TkAgg')

import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import count

from src.train import get_parser
from src.context.sampling import sample_contexts
from src.envs import *

from src.eval.eval_models import setup_env, load_model
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG, PPO, A2C, DQN

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    parser = get_parser()

    outdir = "results/base_vs_context/box2d/CARLLunarLanderEnv/0.5_changingcontextvisible"
    n_eval_eps = 10
    num_contexts = 100
    model_fnames = []
    for root, dirs, filenames in os.walk(outdir):
        for filename in filenames:
            if "rl_model" in filename:
                model_fnames.append(os.path.join(root, filename))
    model_fnames = [m for m in model_fnames if "model" in m]
    model_fnames = ["/home/benjamin/Dokumente/code/tmp/carl/src/results/base_vs_context/classic_control/CARLPendulumEnv/0.25_contexthidden/g/DDPG_3/models/rl_model_850000_steps.zip"]

    k_ep_rew_mean = "ep_rew_mean"
    k_ep_rew_std = "ep_rew_std"
    vec_env_class = None  # DummyVecEnv

    data = []
    contexts_in = {}
    base_context = {'max_speed': 8.0, 'dt': 0.05, 'g': 10, 'm': 1.0, 'l': 1.0}
    gravities = np.linspace(5, 15, 5)
    for i, g in enumerate(gravities):
        c = base_context.copy()
        c['g'] = g
        contexts_in[i] = c

    for i, model_fname in enumerate(model_fnames):
        msg = f"Eval {i+1}/{len(model_fnames)}: {model_fname}."
        print(msg)
        model_fname = Path(model_fname)
        step = -1
        if "rl_model" in model_fname.stem:
            step = int(model_fname.stem.split("_")[-2])
        env = setup_env(path=model_fname.parent, contexts=contexts_in, vec_env_class=vec_env_class)
        model, info = load_model(model_fname)
        train_seed = info['seed']
        context_features = info['context_features']
        n_episodes = len(env.contexts)

        deterministic = True
        steps, rewards = [], []
        contexts = []
        trajectories = []
        for episode in range(n_episodes):
            episode_step, episode_reward = 0, 0

            state = env.reset()
            context = env.context
            trajectory = []
            for _ in count():
                action, _ = model.predict(state, deterministic=deterministic)
                new_state, reward, done, info = env.step(action)
                trajectory.append(new_state)
                episode_reward += reward
                episode_step += 1
                if done:
                    break
                state = new_state
            steps.append(episode_step)
            rewards.append(episode_reward)
            contexts.append(context)
            trajectories.append(trajectory)
        # mean_reward, std_reward = evaluate_policy(
        #     model,
        #     env,  # model.get_env(),
        #     n_eval_episodes=n_eval_eps,
        #     return_episode_rewards=True
        # )
        # D = pd.Series({
        #     k_ep_rew_mean: np.mean(mean_reward),
        #     k_ep_rew_std: np.mean(std_reward),
        #     "train_seed": train_seed,  # [train_seed] * n_eval_eps,
        #     "model_fname": model_fname,
        #     # "context_features": context_features,
        #     "step": step,  # [step] * n_eval_eps,
        #     "n_episodes": n_eval_eps,
        # })
        # data.append(D)

    # if len(model_fnames) > 1:
    #     save_path = os.path.commonpath(model_fnames)
    # else:
    #     p = model_fnames[0]
    #     save_path = p.split("DQN")[0]  # TODO make dynamic
    # save_path = Path(save_path) / "eval_train.csv"
    # df = pd.DataFrame(data)
    # df.to_csv(save_path)

        trajectories = np.array(trajectories)

        plt.ion()
        figsize = (6, 4)
        dpi = 200
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        colors = sns.color_palette("colorblind", n_episodes)

        for j in range(len(contexts)):
            context = contexts[j]
            label = f"g={context['g']:.4f}"
            color = colors[j]
            # X = trajectories[j, :, 0]
            # Y = trajectories[j, :, 1]
            # Z = trajectories[j, :, 2]
            # ax.scatter(X, Y, Z, marker='o', color=color, label=label)
            X = trajectories[j, :, 0]
            # X = np.arccos(X)
            Y = trajectories[j, :, 1]
            # Y = np.arcsin(Y)
            Z = trajectories[j, :, 2]
            ax.scatter(X, Z, color=color, label=label)
        ax.legend()

        fig.set_tight_layout(True)
        plt.show()
