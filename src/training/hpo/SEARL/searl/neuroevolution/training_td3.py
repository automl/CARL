import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(name):
    if name == "adam":
        return torch.optim.Adam
    elif name == "adadelta":
        return torch.optim.Adadelta
    elif name == "adamax":
        return torch.optim.Adamax
    elif name == "rmsprop":
        return torch.optim.RMSprop
    elif name == "sdg":
        return torch.optim.SGD


class TD3Training():

    def __init__(self, config, replay_sample_queue):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.training)
        self.replay_sample_queue = replay_sample_queue

        self.args = config.rl

    @staticmethod
    def update_parameters(indi, replay_sample_queue, iterations):
        args = indi.rl_config
        gamma = args.gamma
        tau = args.tau
        Opti = get_optimizer(args.optimizer)

        actor = indi.actor
        actor_target = type(actor)(**actor.init_dict)
        actor_target.load_state_dict(actor.state_dict())
        actor.to(device)
        actor.train()
        actor_target.to(device)
        actor_optim = Opti(actor.parameters(), lr=args.lr_actor)

        critic_1 = indi.critic_1
        critic_1_target = type(critic_1)(**critic_1.init_dict)
        critic_1_target.load_state_dict(critic_1.state_dict())
        critic_1.to(device)
        critic_1.train()
        critic_1_target.to(device)
        critic_1_optim = Opti(critic_1.parameters(), lr=args.lr_critic)

        critic_2 = indi.critic_2
        critic_2_target = type(critic_2)(**critic_2.init_dict)
        critic_2_target.load_state_dict(critic_2.state_dict())
        critic_2.to(device)
        critic_2.train()
        critic_2_target.to(device)
        critic_2_optim = Opti(critic_2.parameters(), lr=args.lr_critic)

        for it in range(iterations):

            transistion_list = replay_sample_queue.get()

            state_list = []
            action_batch = []
            next_state_batch = []
            reward_batch = []
            done_batch = []
            for transition in transistion_list:
                state_list.append(torch.Tensor(transition.state))
                action_batch.append(torch.tensor(transition.action, dtype=torch.float))
                next_state_batch.append(torch.Tensor(transition.next_state))
                reward_batch.append(torch.Tensor(transition.reward))
                done_batch.append(torch.Tensor(transition.done))

            state_batch = torch.stack(state_list, dim=0)
            action_batch = torch.stack(action_batch, dim=0)
            next_state_batch = torch.stack(next_state_batch, dim=0)
            reward_batch = torch.stack(reward_batch, dim=0)
            done_batch = torch.stack(done_batch, dim=0)

            state = state_batch.to(device)
            action = action_batch.to(device)
            reward = reward_batch.to(device)
            done = 1 - done_batch.to(device)
            next_state = next_state_batch.to(device)

            with torch.no_grad():
                noise = (torch.randn_like(action) * args.td3_policy_noise).clamp(-args.td3_noise_clip,
                                                                                 args.td3_noise_clip)
                next_action = (actor_target(next_state) + noise).clamp(-1, 1)
                print(next_state.size(), next_action.size())
                target_Q1 = critic_1_target(torch.cat([next_state, next_action], 1))
                target_Q2 = critic_2_target(torch.cat([next_state, next_action], 1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * gamma * target_Q)

            current_Q1 = critic_1(torch.cat([state, action], 1))
            current_Q2 = critic_2(torch.cat([state, action], 1))

            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            critic_1_optim.zero_grad()
            critic_loss_1.backward()
            for p in critic_1.parameters():
                p.grad.data.clamp_(max=args.clip_grad_norm)
            critic_1_optim.step()

            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            critic_2_optim.zero_grad()
            critic_loss_2.backward()
            for p in critic_2.parameters():
                p.grad.data.clamp_(max=args.clip_grad_norm)
            critic_2_optim.step()

            if it % args.td3_update_freq == 0:
                actor_loss = -critic_1(torch.cat([state, actor(state)], 1))
                actor_loss = torch.mean(actor_loss)

                actor_optim.zero_grad()
                actor_loss.backward()
                for p in actor.parameters():
                    p.grad.data.clamp_(max=args.clip_grad_norm)
                actor_optim.step()

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(critic_1.parameters(), critic_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(critic_2.parameters(), critic_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        actor_optim.zero_grad()
        critic_1_optim.zero_grad()
        if indi.td3_double_q:
            critic_2_optim.zero_grad()

        indi.actor = actor.cpu().clone()
        indi.critic_1 = critic_1.cpu().clone()
        if indi.td3_double_q:
            indi.critic_2 = critic_2.cpu().clone()
        indi.train_log['train_iterations'] = iterations
        indi.train_log.update(args.get_dict)

        return indi

    def train(self, population, eval_frames, pool=None):
        pop_id_lookup = [ind.index for ind in population]
        iterations = max(self.cfg.train.min_train_steps, int(self.cfg.rl.train_frames_fraction * eval_frames))

        if self.cfg.nevo.ind_memory:
            args_list = [(indi, indi.replay_memory, iterations) for indi in population]
        else:
            args_list = [(indi, self.replay_sample_queue, iterations) for indi in population]

        result_dicts = [pool.apply_async(self.update_parameters, args) for args in args_list]
        trained_pop = [res.get() for res in result_dicts]
        trained_pop = sorted(trained_pop, key=lambda i: pop_id_lookup.index(i.index))

        return trained_pop
