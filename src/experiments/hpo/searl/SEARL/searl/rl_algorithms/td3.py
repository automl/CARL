import copy

import gym
import numpy as np
import torch
import torch.nn.functional as F

from searl.neuroevolution.components.envolvable_mlp import EvolvableMLP
from searl.neuroevolution.training_td3 import get_optimizer
from searl.neuroevolution.components.utils import to_tensor, Transition
from searl.rl_algorithms.components.replay_memory import ReplayBuffer
from searl.utils.supporter import Supporter


class TD3(object):

    def __init__(self, config, logger, checkpoint):

        self.cfg = config
        self.log = logger
        self.ckp = checkpoint

        self.lr_rate = 0.001

        self.env = gym.make(self.cfg.env.name)
        self.cfg.set_attr("action_dim", self.env.action_space.shape[0])
        self.cfg.set_attr("state_dim", self.env.observation_space.shape[0])

        # Set seeds
        self.env.seed(seed=self.cfg.seed.env)
        torch.manual_seed(self.cfg.seed.torch)
        np.random.seed(self.cfg.seed.numpy)

        self.actor = EvolvableMLP(num_inputs=self.cfg.state_dim, num_outputs=self.cfg.action_dim,
                                  **self.cfg.actor.get_dict)

        self.actor_target = type(self.actor)(**self.actor.init_dict)
        self.actor_target.load_state_dict(self.actor.state_dict())

        critic_1_config = copy.deepcopy(self.cfg.critic.get_dict)
        self.critic_1 = EvolvableMLP(num_inputs=self.cfg.state_dim + self.cfg.action_dim, num_outputs=1,
                                     **critic_1_config)
        self.critic_1_target = type(self.critic_1)(**self.critic_1.init_dict)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        if self.cfg.td3.double_q:
            critic_2_config = copy.deepcopy(self.cfg.critic.get_dict)
            self.critic_2 = EvolvableMLP(num_inputs=self.cfg.state_dim + self.cfg.action_dim, num_outputs=1,
                                         **critic_2_config)
            self.critic_2_target = type(self.critic_2)(**self.critic_2.init_dict)
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        Opti = get_optimizer(self.cfg.td3.optimizer)
        self.actor_optim = Opti(self.actor.parameters(), lr=self.cfg.td3.lr_actor)
        self.critic_1_optim = Opti(self.critic_1.parameters(), lr=self.cfg.td3.lr_critic)
        if self.cfg.td3.double_q:
            self.critic_2_optim = Opti(self.critic_2.parameters(), lr=self.cfg.td3.lr_critic)

        self.replay_memory = ReplayBuffer(self.cfg.td3.rm_capacity)

        self.log.print_config(self.cfg)

    def evaluate_policy(self, eval_episodes):
        episode_reward_list = []
        for _ in range(eval_episodes):

            state = self.env.reset()
            t_state = to_tensor(state).unsqueeze(0)
            done = False
            episode_reward = 0
            while not done:
                # Reset environment
                action = self.actor(t_state)
                action.clamp(-1, 1)  # only for MuJoCo
                action = action.data.numpy()
                action = action.flatten()

                step_action = (action + 1) / 2  # [-1, 1] => [0, 1]
                step_action *= (self.env.action_space.high - self.env.action_space.low)
                step_action += self.env.action_space.low

                next_state, reward, done, info = self.env.step(step_action)  # Simulate one step in environment
                t_state = to_tensor(next_state).unsqueeze(0)
                episode_reward += reward

            episode_reward_list.append(episode_reward)

        avg_reward = np.mean(episode_reward_list)

        self.log("---------------------------------------")
        self.log("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        self.log("---------------------------------------")

        return avg_reward

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def perform_learning(self):

        self.log("START LEARNING")

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        while total_timesteps < self.cfg.td3.max_timesteps:

            if done:

                if total_timesteps != 0:
                    self.log("Start Training: Total timesteps: %d Episode Num: %d Episode T: %d Reward: %f" % (
                        total_timesteps, episode_num, episode_timesteps, episode_reward), time_step=total_timesteps)
                    self.log("episode_reward", episode_reward, time_step=total_timesteps)
                    self.train(episode_timesteps, reinit_optim=self.cfg.td3.recreate_optim,
                               reinit_target=self.cfg.td3.reset_target, lr_rate=self.lr_rate)
                # Evaluate episode
                if timesteps_since_eval >= self.cfg.td3.eval_freq:
                    timesteps_since_eval %= self.cfg.td3.eval_freq
                    test_mean_reward = self.evaluate_policy(eval_episodes=self.cfg.td3.eval_episodes)
                    self.log("test_mean_reward", test_mean_reward, time_step=total_timesteps)

                    if self.cfg.support.save_models:
                        self.ckp.save_object(self.actor.state_dict(), name="actor_state_dict")
                        self.ckp.save_object(self.critic_1.state_dict(), name="critic_1_state_dict")

                # Reset environment
                state = self.env.reset()
                t_state = to_tensor(state).unsqueeze(0)
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Select action randomly or according to policy
            if total_timesteps < self.cfg.td3.start_timesteps:
                action = self.env.action_space.sample()
                action = to_tensor(action)
            else:
                action = self.actor(t_state)
            action.clamp(-1, 1)  # only for MuJoCo
            action = action.data.numpy()
            if self.cfg.td3.exploration_noise is not False:
                action += self.cfg.td3.exploration_noise * np.random.randn(self.cfg.action_dim)
                action = np.clip(action, -1, 1)
            action = action.flatten()

            step_action = (action + 1) / 2  # [-1, 1] => [0, 1]
            step_action *= (self.env.action_space.high - self.env.action_space.low)
            step_action += self.env.action_space.low

            next_state, reward, done, info = self.env.step(step_action)  # Simulate one step in environment

            done_bool = 0 if episode_timesteps + 1 == self.env._max_episode_steps else float(done)

            t_next_state = to_tensor(next_state).unsqueeze(0)

            transition = Transition(state, action, next_state, np.array([reward]),
                                    np.array([done_bool]).astype('uint8'))
            self.replay_memory.add(transition)

            t_state = t_next_state
            state = next_state

            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # Final evaluation
        self.log("training end", time_step=total_timesteps)
        test_mean_reward = self.evaluate_policy(eval_episodes=self.cfg.td3.eval_episodes)
        self.log("test_mean_reward", test_mean_reward, time_step=total_timesteps)
        if self.cfg.support.save_models:
            self.ckp.save_state_dict(self.actor.state_dict(), number=1)
            self.ckp.save_state_dict(self.critic_1.state_dict(), number=2)
            if self.cfg.td3.double_q:
                self.ckp.save_state_dict(self.critic_2.state_dict(), number=3)
            self.ckp.save_object(self.replay_memory.storage, name="er_memory")

    def train(self, iterations, reinit_target=False, reinit_optim=False, lr_rate=0.001):

        if reinit_target:
            self.actor_target = type(self.actor)(**self.actor.init_dict)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic_1_target = type(self.critic_1)(**self.critic_1.init_dict)
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())

            self.critic_2_target = type(self.critic_2)(**self.critic_2.init_dict)
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        if reinit_optim:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_rate)
            self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(), lr=lr_rate)
            self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(), lr=lr_rate)

        for it in range(iterations):

            transition_list = self.replay_memory.sample(self.cfg.td3.batch_size)

            state_list = []
            action_batch = []
            next_state_batch = []
            reward_batch = []
            done_batch = []
            indexes = []
            for transition in transition_list:
                state_list.append(torch.Tensor(transition.state))
                action_batch.append(torch.Tensor(transition.action))
                next_state_batch.append(torch.Tensor(transition.next_state))
                reward_batch.append(torch.Tensor(transition.reward))
                done_batch.append(torch.Tensor(transition.done))
                indexes.append(transition.index)

            state = torch.stack(state_list, dim=0)
            action = torch.stack(action_batch, dim=0)
            next_state = torch.stack(next_state_batch, dim=0)
            reward = torch.stack(reward_batch, dim=0)
            done = 1 - torch.stack(done_batch, dim=0)

            with torch.no_grad():
                noise = (torch.randn_like(action) * self.cfg.td3.td3_policy_noise).clamp(-self.cfg.td3.td3_noise_clip,
                                                                                         self.cfg.td3.td3_noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
                target_Q1 = self.critic_1_target(torch.cat([next_state, next_action], 1))
                target_Q2 = self.critic_2_target(torch.cat([next_state, next_action], 1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * self.cfg.td3.gamma * target_Q)

            current_Q1 = self.critic_1(torch.cat([state, action], 1))
            current_Q2 = self.critic_2(torch.cat([state, action], 1))

            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optim.zero_grad()
            critic_loss_1.backward()
            for p in self.critic_1.parameters():
                p.grad.data.clamp_(max=self.cfg.td3.clip_grad_norm)
            self.critic_1_optim.step()

            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optim.zero_grad()
            critic_loss_2.backward()
            for p in self.critic_2.parameters():
                p.grad.data.clamp_(max=self.cfg.td3.clip_grad_norm)
            self.critic_2_optim.step()

            if it % self.cfg.td3.td3_update_freq == 0:

                actor_loss = -self.critic_1(torch.cat([state, self.actor(state)], 1))

                actor_loss = torch.mean(actor_loss)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                for p in self.actor.parameters():
                    p.grad.data.clamp_(max=self.cfg.td3.clip_grad_norm)
                self.actor_optim.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.cfg.td3.tau * param.data + (1 - self.cfg.td3.tau) * target_param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(self.cfg.td3.tau * param.data + (1 - self.cfg.td3.tau) * target_param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(self.cfg.td3.tau * param.data + (1 - self.cfg.td3.tau) * target_param.data)


def start_TD3_training(config, expt_dir):
    with Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True) as sup:
        cfg = sup.get_config()
        log = sup.get_logger()

        env = gym.make(cfg.env.name)
        cfg.set_attr("action_dim", env.action_space.shape[0])
        cfg.set_attr("state_dim", env.observation_space.shape[0])

        td3 = TD3(config=cfg, logger=log, checkpoint=sup.ckp)

        td3.perform_learning()
