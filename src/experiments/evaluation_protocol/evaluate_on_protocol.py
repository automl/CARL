import sys
# FIXME does not run
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("/home/benjamin/Dokumente/code/tmp/CARL/src")
sys.path.append("/home/benjamin/Dokumente/code/tmp/CARL")
print(sys.path)

import hydra
from omegaconf import DictConfig

from pathlib import Path
import numpy as np
import warnings

from src.context.sampling import sample_contexts
from src.train import get_env
from src.experiments.evaluation_protocol.evaluation_protocol_utils import merge_contexts, get_ep_contexts
from src.utils.json_utils import lazy_json_load
from src.training.eval_policy import evaluate_policy


def rollout(cfg):
    context_distribution_type = cfg.context_distribution_type
    n_eval_eps_per_context = cfg.n_eval_eps_per_context
    model_fname = cfg.model_fname

    # Get model setup
    model_fname = Path(model_fname)
    assert model_fname.name == "model.zip"
    setup_fn = model_fname.parent / "trial_setup.json"
    setup = lazy_json_load(setup_fn)
    mode = setup["evaluation_protocol_mode"]
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
    contexts_dict = get_ep_contexts(env_name=env_name, seed=seed, n_contexts=n_contexts, mode=mode)
    contexts = sample_contexts(env_name, [], n_contexts)
    contexts_test_ep = contexts_dict[context_distribution_type]
    contexts_test = merge_contexts(ep_contexts=contexts_test_ep, contexts=contexts)
    if contexts_test is None:
        warnings.warn(f"Selected context distribution type {context_distribution_type} not available in evaluation "
                         f"protocol {mode}. Exiting.")
        return

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
        vec_env_cls=None,
        return_eval_env=False,
    )

    # Load model
    agent_str = setup["agent"]
    agent = eval(agent_str)
    model = agent.load(path=str(model_fname))

    # Setup IO
    output_dir = model_fname.parent / "eval" / "evaluation_protocol"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_fn = output_dir / f"evaluation_on_{context_distribution_type}.npz"

    evaluations_timesteps = []
    evaluations_results = []
    evaluations_length = []
    evaluations_instances = []

    n_eval_episodes = n_contexts * n_eval_eps_per_context
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
        context_distribution_type=context_distribution_type,
    )


@hydra.main("configs", "config_evprot")
def main(cfg: DictConfig):
    rollout(cfg=cfg)


if __name__ == '__main__':
    main()

    # get model fnames
    # path = "/home/benjamin/Dokumente/code/tmp/CARL/src/results/evaluation_protocol/base_vs_context/classic_control/CARLCartPoleEnv"
    # import glob
    # import os
    # model_fnames = glob.glob(os.path.join(path, "**", "model.zip"), recursive=True)
    # string = ",".join(model_fnames)






