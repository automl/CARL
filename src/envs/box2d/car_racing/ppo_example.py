from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# For CarRacing-v0 we can use the hypers from rl-baselines3-zoo
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/0a853b6772e9988e196994a0dcb6b13558158da0/hyperparams/ppo.yml#L361


def mywrap(env):
    """Helper to properly wrap the gym environment before make_vec_env"""
    return GrayScaleObservation(ResizeObservation(env, 64), keep_dim=True)


env = make_vec_env('CarRacing-v0',
                   wrapper_class=mywrap) # ,
                   #n_envs=8)  # This is from rlbaselines3-zoo but takes way too much memory for my local machine
env = VecFrameStack(env, n_stack=4)

model = PPO(CnnPolicy, env, verbose=1,
            batch_size=128, n_steps=512, gamma=.99,
            gae_lambda=0.9, n_epochs=20, ent_coef=0.0, sde_sample_freq=4,
            max_grad_norm=0.5, vf_coef=0.5, learning_rate=3e-5,
            use_sde=True, clip_range=0.4,
            policy_kwargs={'log_std_init': -2,
                           'ortho_init': False})
model.learn(total_timesteps=int(1e6))
model.save("ppo_car")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_car")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
