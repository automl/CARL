from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import os
import json
import argparse
from functools import partial
from src.train import get_parser
from src.context.sampling import sample_contexts

import importlib
import src.training.trial_logger
importlib.reload(src.training.trial_logger)
from src.training.trial_logger import TrialLogger

def setup_agent(config, outdir, parser, args):
    env_wrapper = None
    env = config["env_config"]["env"]
    config["seed"] = config["env_config"]["seed"]
    seed = config["env_config"]["seed"]
    hide_context = config["env_config"]["hide_context"]
    context_args = config["env_config"]["context_args"]
    del config["env_config"]
    timesteps = 0
    config["seed"] = seed

    num_contexts = 100
    contexts = sample_contexts(
        env,
        context_args,
        num_contexts,
        default_sample_std_percentage=0.1
    )
    
    args.agent = "PPO"
    logger = TrialLogger(
        outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=False,
        init_sb3_tensorboard=False  # set to False if using SubprocVecEnv
    )

    train_args_fname = os.path.join(logger.logdir, "trial_setup.json")
    with open(train_args_fname, 'w') as file:
        json.dump(args.__dict__, file, indent="\t")

    contexts_fname = os.path.join(logger.logdir, "contexts_train.json")
    with open(contexts_fname, 'w') as file:
        json.dump(contexts, file, indent="\t")

    env_logger = logger
    from src.envs import CARLPendulumEnv, CARLAcrobotEnv, CARLLunarLanderEnv
    EnvCls = partial(
        eval(env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrapper)
    
    model = PPO('MlpPolicy', env, **config)
    model.set_logger(logger.stable_baselines_logger)
    return model, timesteps, context_args, hide_context

def eval_model(model, eval_env):
    eval_reward = 0
    for i in range(100):
        done = False
        state = eval_env.reset()
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = eval_env.step(action)
            eval_reward += reward
    return eval_reward/100

def step(model, timesteps, env, context_args, hide_context):
    model.learn(4096, reset_num_timesteps=False)
    timesteps += 4096
    num_contexts = 100
    contexts = sample_contexts(
        env,
        context_args,
        num_contexts,
        default_sample_std_percentage=0.1
    )
    env_logger = None
    from src.envs import CARLPendulumEnv, CARLAcrobotEnv, CARLLunarLanderEnv
    EnvCls = partial(
        eval(env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=None)
    eval_reward = eval_model(model, eval_env)
    return eval_reward, model, timesteps

def load_hps(policy_file):
    raw_policy = []
    with open(policy_file, "rt") as fp:
        for row in fp.readlines():
            parsed_row = json.loads(row)
            raw_policy.append(tuple(parsed_row))

    policy = []
    last_new_tag = None
    last_old_conf = None
    for (old_tag, new_tag, old_step, new_step, old_conf, new_conf) in reversed(raw_policy):
        if last_new_tag and old_tag != last_new_tag:
            break
        last_new_tag = new_tag
        last_old_conf = old_conf
        policy.append((new_step, new_conf))

    return last_old_conf, iter(list(reversed(policy)))

parser = argparse.ArgumentParser()
parser.add_argument(
        "--policy_path", help="Path to PBT policy")
parser.add_argument("--seed", type=int)
parser.add_argument("--env", type=str)
parser.add_argument("--hide_context", action='store_true')
parser.add_argument("--context_feature", type=str)
parser.add_argument("--net_size", type=int, default=64)
parser.add_argument("--outdir", type=str)
args, _ = parser.parse_known_args()

env_config = {"seed": args.seed, "env": args.env, "hide_context": args.hide_context, "context_args": [args.context_args]}

config, hp_schedule = load_hps(args.policy_path)
config["env_config"] = env_config
model, timesteps, context_args, hide_context = setup_agent(config, args.outdir, parser, args)
change_at, next_config = next(hp_schedule, None)
for i in range(250):
    config["policy_kwargs"] = {"net_arch": [args.net_size, args.net_size]}
    reward, model, timesteps = step(model, timesteps, args.env, context_args, hide_context)
    print(f"Step: {i*4096}, reward: {reward}")
    if i == change_at:
        model.learning_rate = next_config["learning_rate"]
        model.gamma = next_config["gamma"]
        model.ent_coef = next_config["ent_coef"]
        model.vf_coef = next_config["vf_coef"]
        model.gae_lambda = next_config["gae_lambda"]
        model.max_grad_norm = next_config["max_grad_norm"]
        try:
            change_at, next_config = next(hp_schedule, None)
        except:
            pass
