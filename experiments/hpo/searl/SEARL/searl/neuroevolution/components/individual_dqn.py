import copy

from .envolvable_cnn import EvolvableCnnDQN


class DQNIndividual():

    def __init__(self, state_dim, action_dim, actor_config, rl_config, index, device='cpu', replay_memory=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_config = actor_config
        self.rl_config = rl_config
        self.index = index
        self.device = device

        self.actor = EvolvableCnnDQN(input_shape=state_dim, num_actions=action_dim, device=device,
                                     **actor_config).to(device)

        self.fitness = []
        self.improvement = 0
        self.train_log = {"pre_fitness": None, "pre_rank": None, "post_fitness": None, "post_rank": None, "eval_eps": 0,
                          "index": None, "parent_index": None, "mutation": None}

        self.replay_memory = replay_memory

    def clone(self, index=None):
        if index is None:
            index = self.index

        clone = type(self)(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           actor_config=copy.deepcopy(self.actor.short_dict),
                           rl_config=copy.deepcopy(self.rl_config),
                           index=index,
                           replay_memory=self.replay_memory,
                           device=self.device
                           )

        clone.fitness = copy.deepcopy(self.fitness)
        clone.train_log = copy.deepcopy(self.train_log)
        clone.actor = self.actor.clone()

        if self.replay_memory:
            self.replay_memory = copy.deepcopy(self.replay_memory)

        return clone
