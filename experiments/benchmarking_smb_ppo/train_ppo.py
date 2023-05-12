import atexit
import logging
import os
import time

import hydra
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
from hydra.core.hydra_config import HydraConfig
from modules.ppo import PPOAgent
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from utils.context_utils import get_contexts
from utils.env_utils import make_env
from utils.evaluate import evaluate
from utils.experiment_utils import (
    obs_to_tensor,
    set_seed,
)
from pyvirtualdisplay.display import Display

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="ppo", version_base=None)
def main(cfg: DictConfig):
    logger.info("Training PPO agent")
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

    run_name = f"{cfg.env.id}__{cfg.seed}__{int(time.time())}"

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="train",
        name=run_name,
        save_code=True,
        mode="offline" if cfg.wandb.offline else "online",
    )
    run.log_code(HydraConfig.get().runtime.cwd)

    train_contexts, eval_contexts = get_contexts(cfg), get_contexts(cfg, is_eval=True)
    logger.info(f"Train contexts: {train_contexts}")
    logger.info(f"Eval contexts: {eval_contexts}")

    envs = make_env(**cfg.env, contexts=train_contexts, carl_kwargs=cfg.carl)
    eval_env = make_env(
        id=cfg.eval_env.id,
        num_envs=1,
        capture_video=cfg.eval_env.capture_video,
        visual=True,
        sticky_action_probability=0.0,
        contexts=eval_contexts,
        carl_kwargs=cfg.carl
    )
    
    logger.info(f"Observation space: {envs.single_observation_space}")
    logger.info(f"Action space: {envs.single_action_space}")

    def close_all():
        envs.close()
        eval_env.close()

    atexit.register(close_all)

    agent = PPOAgent(envs).to(device)
    if cfg.carl.hide_context:
        summary(
            agent, (cfg.minibatch_size, *envs.single_observation_space.shape), device=device
        )
    
    optimizer = optim.Adam(agent.parameters(), lr=cfg.optimizer.lr)

    if cfg.restore is not None:
        checkpoint = torch.load(cfg.restore)
        optimizer.load_state_dict(checkpoint["optimizer"])
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
        },
        step=global_step,
    )

    if cfg.carl.hide_context:
        obs = torch.zeros(
            (cfg.num_steps, cfg.env.num_envs) + envs.single_observation_space.shape
        ).to(device)
    else:
        obs = torch.zeros(
            (
                cfg.num_steps,
                cfg.env.num_envs
            ) + envs.single_observation_space.spaces["state"].shape
        ).to(device)
        contexts = torch.zeros(
            (
                cfg.num_steps,
                cfg.env.num_envs
            ) + envs.single_observation_space.spaces["context"].shape
        ).to(device)
    actions = torch.zeros(
        (cfg.num_steps, cfg.env.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.env.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.env.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.env.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.env.num_envs)).to(device)

    start_time = time.time()
    num_updates = cfg.total_timesteps // cfg.batch_size

    next_obs = obs_to_tensor(envs.reset(seed=cfg.seed), device)
    next_done = torch.zeros(cfg.env.num_envs).to(device)

    for update in range(1, num_updates + 1):
        for step in range(0, cfg.num_steps):
            global_step += 1 * cfg.env.num_envs
            if cfg.carl.hide_context:
                obs[step] = next_obs
            else:
                obs[step] = next_obs["state"]
                contexts[step] = next_obs["context"]
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, info = envs.step(
                action.cpu().numpy()
            )
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = obs_to_tensor(next_obs, device), torch.Tensor(
                done
            ).to(device)
            done = np.array(done)

            if "episode" in info and done.any():
                logger.info(
                    f"step={global_step}, episode_reward={info['episode']['r'][done].mean()}"
                )
                final_infos = [i for i in info["terminal_info"][done]]
                completed_info = [i["completed"] for i in final_infos]
                wandb.log(
                    {
                        "train/episode_reward": info["episode"]["r"][done].mean(),
                        "train/episode_length": info["episode"]["l"][done].mean(),
                        "train/episode_reward_hist": wandb.Histogram(
                            info["episode"]["r"][done]
                        ),
                        "train/episode_length_hist": wandb.Histogram(
                            info["episode"]["l"][done]
                        ),
                        "train/episode_completed": np.mean(completed_info),
                        "train/episode_completed_hist": wandb.Histogram(completed_info),
                    },
                    step=global_step,
                )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        if cfg.carl.hide_context:
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        else:
            b_obs = obs.reshape((-1,) + envs.single_observation_space.spaces["state"].shape)
            b_contexts = contexts.reshape((-1,) + envs.single_observation_space.spaces["context"].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.long().reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                if cfg.carl.hide_context:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                else: 
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        dict(state=b_obs[mb_inds], context=b_contexts[mb_inds]), b_actions[mb_inds]
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.normalize_advantage:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_value_loss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_ratio,
                        cfg.clip_ratio,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - cfg.entropy_weight * entropy_loss
                    + v_loss * cfg.value_weight
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl_div is not None:
                if approx_kl > cfg.target_kl_div:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log(
            {
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/fps": int(global_step / (time.time() - start_time)),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
            },
            step=global_step,
        )

        eval_return, eval_completed = evaluate(agent, eval_env, cfg)
        wandb.log(
            {
                "eval/episode_reward": eval_return,
                "eval/episode_completed": eval_completed,
            },
            step=global_step,
        )
        checkpoint_path = os.path.join(wandb.run.dir, "agent.pt")
        torch.save(
            {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            },
            checkpoint_path,
        )
        wandb.save(checkpoint_path)


if __name__ == "__main__":
    with Display() as disp:
        main()
