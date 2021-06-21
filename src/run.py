import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
# from classic_control.meta_mountaincar import MountainCarEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv

env_class = "MountainCar-v0"
env_class = MountainCarEnv
env_config = {}

seed = 213234
out_dir = "runs"

logger_config = {"logdir": out_dir}

trainer_class = PPOTrainer
config = {
    "env": env_class,
    "env_config": env_config,
    "log_level": "WARNING",
    "framework": "torch",
    # "seed": seed,
    # "horizon": 1_000_000,
    # "local_dir": out_dir,
    "logger_config": logger_config,
}

# ray.shutdown()
# ray.init()
# trainer = trainer_class(env=env_class, config=config)
# trainer.train()
# ray.shutdown()

tune.run(trainer_class, config=config, local_dir=out_dir)
