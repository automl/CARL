import sys
# FIXME does not run
sys.path.append("")
sys.path.append("../../carl/experiments")
sys.path.append("../../carl")
sys.path.append("../..")
sys.path.append("/home/benjamin/Dokumente/code/tmp/CARL/carl")
sys.path.append("/home/benjamin/Dokumente/code/tmp/CARL")
print(sys.path)

import hydra
from omegaconf import DictConfig

from pathlib import Path
import numpy as np
import warnings
import glob

import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv

from carl.train import get_env
from carl.utils.json_utils import lazy_json_load
from carl.training.eval_policy import evaluate_policy


def rollout(cfg):
    model_fname = cfg.model_fname

    # Get model setup
    model_fname = Path(model_fname)
    if model_fname.is_dir():
        model_fnames = glob.glob(str(model_fname / "rl_model_*.zip"))
        model_fnames.sort()
    else:
        model_fnames = [model_fname]

    for mf in model_fnames:
        print(mf)

    evaluations_timesteps = []
    evaluations_results = []
    evaluations_length = []
    evaluations_instances = []
    for model_fname in model_fnames:
        model_fname = Path(model_fname)
        exp_dir = model_fname.parent if not "rl_model" in str(model_fname) else model_fname.parent.parent
        setup_fn = exp_dir / "trial_setup.json"
        setup = lazy_json_load(setup_fn)
        seed = setup["seed"]
        env_name = setup["env"]
        n_contexts = setup["num_contexts"]

        # # Get Hyperparameters
        # hp_fn = model_fname.parent / "hyperparameters.json"
        # hps = lazy_json_load(hp_fn)
        # hyperparams = hps["hyperparameters"]
        # env_wrapper = hps["env_wrapper"]
        # normalize_kwargs = hps["normalize_kwargs"]
        # schedule_kwargs = hps["schedule_kwargs"]
        # normalize = False
        # if normalize_kwargs is not None and normalize_kwargs["normalize"]:
        #     normalize = True

        # Get contexts
        if cfg.context_file is None:
            context_file = exp_dir / "contexts_train.json"
            evaluate_on_train_contexts = True
        else:
            context_file = cfg.context_file
            evaluate_on_train_contexts = False
        contexts_test = lazy_json_load(context_file)

        # Setup env
        env_kwargs = dict(
            contexts=contexts_test,
            hide_context=setup['hide_context'],
            scale_context_features=setup['scale_context_features'],
            state_context_features=setup['state_context_features']
        )
        env = get_env(
            env_name=env_name,
            n_envs=1,
            env_kwargs=env_kwargs,
            wrapper_class=None,
            wrapper_kwargs = None,
            normalize_kwargs=None,
            agent_cls=None,
            eval_seed=None,
            return_vec_env=True,
            vec_env_cls=DummyVecEnv,
            return_eval_env=False,
        )
        env.seed(seed)

        # Load model
        agent_str = setup["agent"]
        agent = getattr(stable_baselines3, agent_str)
        model = agent.load(path=str(model_fname), seed=seed)

        # Setup IO
        if evaluate_on_train_contexts:
            output_dir = exp_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluation_fn = output_dir / f"evaluations.npz"
        else:
            raise NotImplementedError("Need to define a proper save path")

        # if evaluation_fn.is_file():
        #     data = np.load(str(evaluation_fn))
        #     evaluations_timesteps = list(data["timesteps"])
        #     evaluations_results = list(data["results"])
        #     evaluations_length = list(data["ep_lengths"])
        #     evaluations_instances = list(data["episode_instances"])
        # else:

        n_eval_episodes = n_contexts
        episode_rewards, episode_lengths, episode_instances = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
        )

        evaluations_timesteps.append(model.num_timesteps)
        evaluations_results.append(episode_rewards)
        evaluations_length.append(episode_lengths)
        evaluations_instances.append(episode_instances)

        np.savez(
            str(evaluation_fn),
            timesteps=evaluations_timesteps,
            results=evaluations_results,
            ep_lengths=evaluations_length,
            episode_instances=evaluations_instances,
        )


@hydra.main("configs", "config_eval_models")
def main(cfg: DictConfig):
    rollout(cfg=cfg)


if __name__ == '__main__':
    main()
