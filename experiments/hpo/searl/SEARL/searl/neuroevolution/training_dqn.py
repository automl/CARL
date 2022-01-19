import numpy as np
import torch

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


class DQNTraining():

    def __init__(self, config, replay_memory, replay_priority_queue=None):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.training)
        self.replay_sample_queue = replay_memory
        self.replay_priority_queue = replay_priority_queue
        self.args = config.rl

    @staticmethod
    def update_parameters(indi, replay_sample_queue, iterations):
        args = indi.rl_config
        Opti = get_optimizer(args.optimizer)

        actor = indi.actor
        actor_target = type(actor)(**actor.init_dict)
        actor_target.load_state_dict(actor.state_dict())
        actor.to(device)
        actor.train()
        actor_target.to(device)
        actor_optim = Opti(actor.parameters(), lr=args.lr_actor)

        losses = []
        for it in range(iterations):
            transistion_list = replay_sample_queue.get()
            state_list = []
            action_batch = []
            next_state_batch = []
            reward_batch = []
            done_batch = []
            for transition in transistion_list:
                state_list.append(transition.state)
                action_batch.append(transition.action)
                next_state_batch.append(transition.next_state)
                reward_batch.append(transition.reward)
                done_batch.append(transition.done)

            state = torch.stack(state_list, dim=0).to(device)
            action = torch.stack(action_batch, dim=0).squeeze().to(device)
            next_state = torch.stack(next_state_batch, dim=0).to(device)
            rewards = torch.stack(reward_batch, dim=0).squeeze().to(device)
            dones = torch.stack(done_batch, dim=0).squeeze().to(device)

            with torch.no_grad():
                batch_size = next_state.size(0)

                delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)
                support = torch.linspace(args.Vmin, args.Vmax, args.num_atoms).to(device)

                next_dist = actor_target(next_state) * support
                next_action = next_dist.sum(2).max(1)[1]
                next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
                next_dist = next_dist.gather(1, next_action).squeeze(1)

                rewards = rewards.unsqueeze(1).expand_as(next_dist)
                dones = dones.unsqueeze(1).expand_as(next_dist)
                support = support.unsqueeze(0).expand_as(next_dist)

                Tz = rewards + (1 - dones) * 0.99 * support
                Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
                b = (Tz - args.Vmin) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = torch.linspace(0, (batch_size - 1) * args.num_atoms, batch_size).long() \
                    .unsqueeze(1).expand(batch_size, args.num_atoms).to(device)

                proj_dist = torch.zeros(next_dist.size()).to(device)
                proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            dist = actor(state)
            action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
            dist = dist.gather(1, action).squeeze(1)
            dist.data.clamp_(0.01, 0.99)
            loss = -(proj_dist * dist.log()).sum(1)
            loss = loss.mean()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()

            if it % 5 == 0:
                actor.reset_noise()
                actor_target.reset_noise()

            losses.append(loss.detach().cpu().numpy())

            if it % 2 == 0 and it != 0:
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        indi.actor = actor.cpu().clone()
        indi.train_log['train_iterations'] = iterations
        indi.train_log['train_losses'] = np.mean(losses).tolist()
        indi.train_log.update(args.get_dict)

        return indi

    def train(self, population, iterations, pool=None):

        pop_id_lookup = [ind.index for ind in population]

        if self.cfg.nevo.ind_memory:
            args_list = [(indi, indi.replay_memory, iterations) for indi in population]
        else:
            args_list = [(indi, self.replay_sample_queue, iterations) for indi in population]

        trained_pop = []
        for args in args_list:
            trained_pop.append(self.update_parameters(*args))

        trained_pop = sorted(trained_pop, key=lambda i: pop_id_lookup.index(i.index))

        return trained_pop
