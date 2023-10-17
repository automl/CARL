from brax.training.agents.ppo import train
import functools
import wandb
from brax.io import model
import os
import cloudpickle
import jax
import numpy as np


def ppo(cfg, env, eval_env):
    train_fn = functools.partial(train.train, num_timesteps=500, num_evals=1, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1) #, **cfg.train_kwargs)
    def progress(num_steps, metrics):
        wandb.log(metrics, commit=False)

    make_inference_fn, params, metrics = train_fn(environment=env, eval_env=eval_env, progress_fn=progress)
    model.save_params(os.path.join(wandb.run.dir, 'params'), params)
    with open(os.path.join(wandb.run.dir, 'make_inference.pkl'), 'wb+') as f:
        cloudpickle.dump(make_inference_fn, f)
    return metrics


def evaluate_ppo(cfg, env):
    with open(cfg.inference_file, "rb") as f:
        inference_fn = cloudpickle.load(f)
    params = model.load_params(cfg.param_file)
    inference_fn = inference_fn(params)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    for _ in range(cfg.num_eval_steps):
        rollout.append(state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
    return np.mean(state.reward)