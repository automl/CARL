import os
from typing import Optional

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# from searl.rl_algorithms.components.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from src.train import get_env, set_hps, get_parser, get_contexts
from src.utils.json_utils import lazy_json_dump


def make_searl_env(env_name: str):
    if env_name == "CARLAnt":
        parser = get_parser()
        args, unknown_args = parser.parse_known_args()
        args.env = "CARLAnt"
        args.state_context_features = "changing_context_features"
        args.hide_context = False
        args.default_sample_std_percentage = 0.1
        args.context_feature_args = ["friction"]
        args.num_envs = 1

        env_wrapper = None
        normalize_kwargs = None

        vec_env_cls = DummyVecEnv
        # Get Contexts
        contexts = get_contexts(args)
        # if logdir is not None:
        #     contexts_fname = os.path.join(logdir, "contexts_train.json")
        #     lazy_json_dump(data=contexts, filename=contexts_fname)

        env_kwargs = dict(
            contexts=contexts,
            logger=None,
            hide_context=args.hide_context,
            scale_context_features=args.scale_context_features,
            state_context_features=args.state_context_features
        )
        env = get_env(
            env_name=args.env,
            n_envs=args.num_envs,
            env_kwargs=env_kwargs,
            wrapper_class=env_wrapper,
            vec_env_cls=vec_env_cls,
            return_eval_env=False,
            normalize_kwargs=normalize_kwargs,
            return_vec_env=False
        )
    else:
        raise NotImplementedError

        # # Atari
        # env = make_atari(env_name)
        # env = wrap_deepmind(env)
        # env = wrap_pytorch(env)
        #
        # # Normal gym
        # env = gym.make(env_name)

    return env
