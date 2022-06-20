import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
import string
import random
from rich import print, inspect

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig

import stable_baselines3
from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.logger import configure

from experiments.context_gating.utils import check_wandb_exists, set_seed_everywhere
from experiments.common.train.utils import make_env
from experiments.common.train.eval_callback import DACEvalCallback
from experiments.common.train.eval_policy import evaluate_policy
from experiments.carlbench.context_logging import log_contexts_wandb
from carl.context.sampling import sample_contexts


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def check_config_valid(cfg):
    valid = True
    if cfg.carl.env_kwargs.hide_context and cfg.carl.env_kwargs.state_context_features:
        valid = False
    if (
        not cfg.contexts.context_feature_args
        and cfg.carl.env_kwargs.state_context_features is not None
    ):
        valid = False
    return valid


@hydra.main("./configs", "base")
def main(cfg: DictConfig):
    sys.path.append(os.getcwd())
    sys.path.append("..")

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if (
        not check_config_valid(cfg)
        or check_wandb_exists(
            dict_cfg,
            unique_fields=[
                "env",
                "seed",
                "group",
                "contexts.context_feature_args",
                "contexts.default_sample_std_percentage",
                "carl.state_context_features",
            ],
        )
    ) and not cfg.debug:
        print(f"Skipping run with cfg {dict_cfg}")
        return

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + id_generator()

    run = wandb.init(
        id=cfg.wandb.id,
        resume="allow",
        mode="offline" if cfg.wandb.debug else None,
        project="carl",
        job_type=cfg.wandb.job_type,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        dir=os.getcwd(),
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
    )
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None
    wandb.config.update({"command": command, "slurm_id": slurm_id})
    set_seed_everywhere(cfg.seed)

    # ----------------------------------------------------------------------
    # Sample Train and Eval Contexts
    # ----------------------------------------------------------------------
    # TODO build in loading context from file
    contexts = sample_contexts(cfg.env, **cfg.contexts)
    if cfg.training.eval_on_train_context:
        eval_contexts = contexts
    else:
        eval_contexts = sample_contexts(cfg.env, **cfg.contexts)
    log_contexts_wandb(train_contexts=contexts, eval_contexts=eval_contexts)
    env = make_env(cfg, contexts=contexts, num_envs=cfg.training.num_envs)

    print(OmegaConf.to_yaml(cfg))
    inspect(env)
    print(f"Observation Space: ", env.observation_space.shape)
    print(f"Action Space: ", env.action_space)

    output_dir = os.getcwd()
    print("Output directory:", output_dir)

    # ----------------------------------------------------------------------
    # Setup logging
    # ----------------------------------------------------------------------
    sb_loggers = ["stdout", "csv", "tensorboard"]
    stable_baselines_logger = configure(str(output_dir), sb_loggers)

    # ----------------------------------------------------------------------
    # Setup callbacks
    # ----------------------------------------------------------------------
    # eval callback actually records performance over all instances while progress writes performance of the last
    # episode(s) which can be a random set of instances
    callbacks = []
    if cfg.training.eval_callback.use:
        eval_env = make_env(cfg, contexts=eval_contexts, num_envs=1)
        n_instances = len(env.instance_set)
        eval_callback = DACEvalCallback(
            eval_env=eval_env,
            log_path=output_dir,
            n_eval_episodes=n_instances,
            eval_freq=1,
            **cfg.training.eval_callback.kwargs,
        )
        everynstep_callback = EveryNTimesteps(
            n_steps=cfg.training.eval_callback.kwargs.eval_freq, callback=eval_callback
        )
        callbacks.append(everynstep_callback)
    callbacks.append(
        WandbCallback(
            verbose=1,
            model_save_path=os.path.join(output_dir, "models"),
            model_save_freq=cfg.training.checkpoint_callback.kwargs.save_freq,
            gradient_save_freq=0,
        )
    )

    # ----------------------------------------------------------------------
    # Create agent
    # ----------------------------------------------------------------------
    agent_cls = getattr(stable_baselines3, cfg.agent.name)
    agent_kwargs = OmegaConf.to_container(cfg=cfg.agent.kwargs, resolve=True)
    agent_kwargs["seed"] = int(agent_kwargs["seed"])
    if "action_noise" in agent_kwargs:
        if "mean" in agent_kwargs["action_noise"]:
            agent_kwargs["action_noise"]["mean"] = np.array(agent_kwargs["action_noise"]["mean"])
        agent_kwargs["action_noise"] = hydra.utils.instantiate(agent_kwargs["action_noise"])
    if cfg.carl.use_cgate:
        agent_kwargs["policy"] = hydra.utils.call(agent_kwargs["policy"])
    if "train_freq" in agent_kwargs:
        agent_kwargs["train_freq"] = tuple(agent_kwargs["train_freq"])
        print(agent_kwargs["train_freq"], agent_kwargs["train_freq"][0], type(agent_kwargs["train_freq"][0]))
    agent_kwargs["tensorboard_log"] = str(output_dir)
    if "learning_rate" in agent_kwargs and type(agent_kwargs["learning_rate"]) != float:
        agent_kwargs["learning_rate"] = hydra.utils.instantiate(agent_kwargs["learning_rate"])
    print(agent_kwargs)
    model = agent_cls(env=env, verbose=2, **agent_kwargs)
    model.set_logger(stable_baselines_logger)

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    model.learn(total_timesteps=cfg.training.num_steps, callback=callbacks)

    # ----------------------------------------------------------------------
    # Save and cleanup
    # ----------------------------------------------------------------------
    model.save(os.path.join(str(output_dir), "model.zip"))
    run.finish()

    # ----------------------------------------------------------------------
    # Final evaluation of model for HPO
    # ----------------------------------------------------------------------
    final_ep_mean_reward = None
    if cfg.job_type == "sweep":
        eval_env = make_env(cfg, contexts=eval_contexts, num_envs=1)
        episode_rewards, episode_lengths, episode_instances = evaluate_policy(
            model=model,
            env=eval_env,
            n_eval_episodes=cfg.contexts.num_contexts,
            deterministic=True,
            render=False,
            return_episode_rewards=True,
            warn=True,
        )
        final_ep_mean_reward = np.mean(episode_rewards)
    return final_ep_mean_reward


if __name__ == "__main__":
    main()
