from stable_baselines3 import PPO

def ppo(cfg, env, eval_env):
    model = PPO("MlpPolicy", env, verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    print("Done")