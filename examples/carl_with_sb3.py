import carl
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from carl.envs import CARLLunarLander
from carl.context.context_space import NormalFloatContextFeature
from carl.context.sampler import ContextSampler

# Create environment
context_distributions = [NormalFloatContextFeature("GRAVITY_X", mu=9.8, sigma=1)]
context_sampler = ContextSampler(
    context_distributions=context_distributions,
    context_space=CARLLunarLander.get_context_space(),
    seed=42,
)
contexts = context_sampler.sample_contexts(n_contexts=5)

print("Training contexts are:")
print(contexts)

env = gym.make("carl/CARLLunarLander-v0", render_mode="rgb_array", contexts=contexts)
env = FlattenObservation(env)

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e4), progress_bar=True)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
