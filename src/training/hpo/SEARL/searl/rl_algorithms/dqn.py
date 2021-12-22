import numpy as np
import torch

from searl.neuroevolution.training_td3 import get_optimizer
from searl.neuroevolution.components.utils import Transition
from searl.neuroevolution.components.envolvable_cnn import EvolvableCnnDQN
from searl.rl_algorithms.components.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from searl.rl_algorithms.components.replay_memory import ReplayBuffer
from searl.utils.supporter import Supporter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA", device == torch.device("cuda"), device)


class DQN(object):

    def __init__(self, config, logger, checkpoint):
        self.cfg = config
        self.log = logger
        self.ckp = checkpoint

        env = make_atari(self.cfg.env.name)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
        self.env = env

        self.cfg.set_attr("action_dim", self.env.action_space.n)
        self.cfg.set_attr("state_dim", self.env.observation_space.shape)

        # Set seeds
        self.env.seed(seed=self.cfg.seed.env)
        torch.manual_seed(self.cfg.seed.torch)
        np.random.seed(self.cfg.seed.numpy)

        self.Vmin = self.cfg.actor.Vmin
        self.Vmax = self.cfg.actor.Vmax
        self.num_atoms = self.cfg.actor.num_atoms
        self.batch_size = self.cfg.dqn.batch_size

        self.tau = 0.005

        self.actor = EvolvableCnnDQN(input_shape=self.cfg.state_dim, num_actions=self.cfg.action_dim, device=device,
                                     **self.cfg.actor.get_dict).to(device)

        Opti = get_optimizer(self.cfg.dqn.optimizer)
        self.actor_optim = Opti(self.actor.parameters(), lr=self.cfg.dqn.lr_actor)

        self.actor_target = type(self.actor)(**self.actor.init_dict).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.replay_memory = ReplayBuffer(self.cfg.dqn.rm_capacity)

        self.log.print_config(self.cfg)

    def evaluate_policy(self, eval_episodes):
        episode_reward_list = []
        for _ in range(eval_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.actor.act(state)
                next_state, reward, done, info = self.env.step(action)  # Simulate one step in environment
                state = next_state
                episode_reward += reward

            episode_reward_list.append(episode_reward)

        avg_reward = np.mean(episode_reward_list)

        self.log("---------------------------------------")
        self.log("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        self.log("---------------------------------------")

        return avg_reward

    def perform_learning(self):

        self.log("START LEARNING")

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_reward = 0
        episode_timesteps = 0
        reset_timesteps = 0
        episode_num = 0
        done = True

        while total_timesteps < self.cfg.dqn.num_frames:

            if done:
                if total_timesteps != 0 and self.replay_memory.storage.__len__() > self.cfg.dqn.replay_initial:
                    if (
                            self.cfg.dqn.reset_target or self.cfg.dqn.recreate_optim) and reset_timesteps >= self.cfg.dqn.min_eval_steps:
                        self.train(episode_timesteps, reinit_optim=self.cfg.dqn.recreate_optim,
                                          reinit_target=self.cfg.dqn.reset_target)
                        reset_timesteps = 0
                    else:
                        self.train(episode_timesteps)

                # Evaluate episode
                if timesteps_since_eval >= self.cfg.dqn.eval_freq:
                    timesteps_since_eval = 0
                    test_mean_reward = self.evaluate_policy(eval_episodes=self.cfg.dqn.eval_episodes)
                    self.log("test_mean_reward", test_mean_reward, time_step=total_timesteps)
                    self.log("test_episode_num", episode_num, time_step=total_timesteps)

                    if self.cfg.support.save_models:
                        self.ckp.save_object(self.actor.state_dict(), name="actor_state_dict")

                # Reset environment
                state = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Select action randomly or according to policy
            if total_timesteps < self.cfg.dqn.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.actor.act(state)

            next_state, reward, done, info = self.env.step(action)  # Simulate one step in environment

            transition = Transition(torch.FloatTensor(state), torch.LongTensor([action]),
                                    torch.FloatTensor(next_state), torch.FloatTensor(np.array([reward])),
                                    torch.FloatTensor(np.array([done]).astype('uint8'))
                                    )
            self.replay_memory.add(transition)

            state = next_state

            episode_reward += reward
            episode_timesteps += 1
            reset_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # Final evaluation
        self.log("training end", time_step=total_timesteps)
        test_mean_reward = self.evaluate_policy(eval_episodes=self.cfg.td3.eval_episodes)
        self.log("test_mean_reward", test_mean_reward, time_step=total_timesteps)
        if self.cfg.support.save_models:
            self.ckp.save_state_dict(self.actor.state_dict(), number=1)
            self.ckp.save_object(self.replay_memory.storage, name="replay_memory")

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)

        next_dist = self.actor_target(next_state) * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms).to(device)

        proj_dist = torch.zeros(next_dist.size()).to(device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def train(self, iterations, reinit_target=False, reinit_optim=False):

        iterations = min(iterations, 10000)

        if reinit_target:
            self.actor_target.load_state_dict(self.actor.state_dict())

        if reinit_optim:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        losses = []

        for it in range(iterations):

            transition_list = self.replay_memory.sample(self.cfg.dqn.batch_size)

            state_list = []
            action_batch = []
            next_state_batch = []
            reward_batch = []
            done_batch = []
            indexes = []
            for transition in transition_list:
                state_list.append(transition.state)
                action_batch.append(transition.action)
                next_state_batch.append(transition.next_state)
                reward_batch.append(transition.reward)
                done_batch.append(transition.done)
                indexes.append(transition.index)

            state = torch.stack(state_list, dim=0).to(device)
            action = torch.stack(action_batch, dim=0).squeeze().to(device)
            next_state = torch.stack(next_state_batch, dim=0).to(device)
            reward = torch.stack(reward_batch, dim=0).squeeze().to(device)
            done = torch.stack(done_batch, dim=0).squeeze().to(device)

            with torch.no_grad():
                proj_dist = self.projection_distribution(next_state, reward, done)

            dist = self.actor(state)
            action = action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
            dist = dist.gather(1, action).squeeze(1)
            dist.data.clamp_(0.01, 0.99)
            loss = -(proj_dist * dist.log()).sum(1)
            loss = loss.mean()

            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

            if it % 5 == 0:
                self.actor.reset_noise()
                self.actor_target.reset_noise()

            losses.append(loss.detach().cpu().numpy())

            if self.cfg.dqn.soft_update:
                if it % 2 == 0 and it != 0:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            else:
                if (it % 1000 == 0 and it != 0) or it == (iterations - 1):
                    self.actor_target.load_state_dict(self.actor.state_dict())
        return np.mean(losses).tolist()


def start_DQN_training(config, expt_dir):
    with Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True) as sup:
        cfg = sup.get_config()
        log = sup.get_logger()

        dqn = DQN(config=cfg, logger=log, checkpoint=sup.ckp)
        dqn.perform_learning()
