import atexit
import logging
import os
import time

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from utils.evaluate import evaluate
from modules.ppo import PPOAgent
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from utils.context_utils import get_contexts
from utils.env_utils import make_env
from utils.experiment_utils import (
    set_seed,
)
from xvfbwrapper import Xvfb

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="ppo", version_base=None)
def main(cfg: DictConfig):
    logger.info("Evaluating PPO agent")
    cfg.batch_size = int(cfg.env.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.device = (
        "cuda:0"
        if torch.cuda.is_available() and cfg.cuda
        else "cpu"
    )
    device = torch.device(cfg.device)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.seed, cfg.torch_deterministic)
    xvfb = Xvfb(tempdir=os.getcwd())
    xvfb.start()

    run_name = f"eval__{cfg.env.id}__{cfg.seed}__{int(time.time())}"

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="eval",
        name=run_name,
        save_code=True,
        mode="offline" if cfg.wandb.offline else "online",
    )
    run.log_code(HydraConfig.get().runtime.cwd)

    eval_contexts = get_contexts(cfg)

    envs = make_env(**cfg.env, contexts=eval_contexts)
    eval_env = make_env(
        id=cfg.eval_env.id,
        num_envs=1,
        capture_video=cfg.eval_env.capture_video,
        visual=True,
        sticky_action_probability=0.0,
        capture_all_episodes=cfg.eval_env.capture_all_episodes,
        contexts=eval_contexts,
    )

    def close_all():
        eval_env.close()
        xvfb.stop()

    atexit.register(close_all)

    agent = PPOAgent(envs).to(device)
    summary(
        agent, (cfg.minibatch_size, *envs.single_observation_space.shape), device=device
    )

    if cfg.restore is not None:
        checkpoint = torch.load(cfg.restore)
        agent.load_state_dict(checkpoint["agent"])
        global_step = checkpoint["global_step"]
        logger.info(
            f"Restored from checkpoint {cfg.restore} after {checkpoint['global_step']} steps"
        )
    else:
        global_step = 0

    eval_return, eval_completed = evaluate(agent, eval_env, cfg)
    wandb.log(
        {
            "eval/episode_reward": eval_return,
            "eval/episode_completed": eval_completed,
            "eval/global_step": global_step,
        },
        step=0,
    )


if __name__ == "__main__":
    main()
