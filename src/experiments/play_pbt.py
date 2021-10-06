from ray import tune
from ray.tune.schedulers import PopulationBasedTrainingReplay
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import os
import yaml
import argparse
from functools import partial

from src.utils.hyperparameter_processing import preprocess_hyperparams
from src.train import get_parser
from src.context.sampling import sample_contexts

class PPOTrainable(tune.Trainable):
    def setup(self, config):
        with open("/home/eimer/Dokumente/git/meta-gym/src/utils/rl_baselines3_zoo/hyperparams/ppo.yml", "r") as f:
            hyperparams_dict = yaml.safe_load(f)
        #hyperparams = hyperparams_dict["AntBulletEnv-v0"]
        #hyperparams, self.env_wrapper, normalize, normalize_kwargs = preprocess_hyperparams(hyperparams)
        hyperparams = {}
        self.env_wrapper = None
        self.env = "CARLAcrobotEnv"
        config["seed"] = 0
        self.timesteps = 0

        num_contexts = 100
        contexts = sample_contexts(
            self.env,
            "friction",
            num_contexts,
            default_sample_std_percentage=0.1
        )
        env_logger = None
        from src.envs import CARLAcrobotEnv
        EnvCls = partial(
            eval(self.env),
            contexts=contexts,
            logger=env_logger,
            hide_context=False,
        )
        env = make_vec_env(EnvCls, n_envs=1, wrapper_class=self.env_wrapper) 

        self.model = PPO('MlpPolicy', env, **config)

    def eval_model(self, eval_env):
        eval_reward = 0
        for i in range(100):
            done = False
            state = eval_env.reset()
            while not done:
                action, _ = self.model.predict(state)
                state, reward, done, _ = eval_env.step(action)
                eval_reward += reward
        return eval_reward/100

    def step(self):
        self.model.learn(4096)
        self.timesteps += 4096
        num_contexts = 100
        contexts = sample_contexts(
            self.env,
            "friction",
            num_contexts,
            default_sample_std_percentage=0.1
            )
        env_logger = None
        from src.envs import CARLAcrobotEnv
        EnvCls = partial(
            eval(self.env),
            contexts=contexts,
            logger=env_logger,
            hide_context=False,
        )
        eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=self.env_wrapper)
        eval_reward = self.eval_model(eval_env)
        return {"ep_reward": eval_reward, "total_timesteps": self.timesteps}

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        self.model.save(path)
        return path

    def load_checkpoint(self, checkpoint_path):
        num_contexts = 100
        contexts = sample_contexts(
                self.env,
                "friction",
                num_contexts,
                default_sample_std_percentage=0.1
            )
        env_logger = None
        from src.envs import CARLAcrobotEnv
        EnvCls = partial(
            eval(self.env),
            contexts=contexts,
            logger=env_logger,
            hide_context=False,
        )
        env = make_vec_env(EnvCls, n_envs=1, wrapper_class=self.env_wrapper)
        
        checkpoint_dir = str(checkpoint_dir)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        self.model = PPO.load(checkpoint, env=env)

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = os.path.join(export_dir, "exported_convnet.pt")
            self.model.save(path)
            return {export_formats[0]: path}
        else:
            raise ValueError("unexpected formats: " + str(export_formats))

    def reset_config(self, new_config):
        self.model.learning_rate = new_config["learning_rate"]
        self.model.gamma = new_config["gamma"]
        self.model.ent_coef = new_config["ent_coef"]
        self.model.vf_coef = new_config["vf_coef"]
        self.model.gae_lambda = new_config["gae_lambda"]
        self.model.max_grad_norm = new_config["max_grad_norm"]
        self.config = new_config
        return True


parser = argparse.ArgumentParser()
parser.add_argument(
        "--policy_path", help="Path to PBT policy")
parser.add_argument("--seed", type=int)
args, _ = parser.parse_known_args()

replay = PopulationBasedTrainingReplay(args.policy_path)

tune.run(
    PPOTrainable,
    name="p7_s0",
    scheduler=replay,
    stop={"training_iteration": 250},
    local_dir=f"/home/eimer/Dokumente/git/meta-gym/src/results/classic_control/pbt_hps/CARLAcrobotEnv/",
    log_to_file=True)
