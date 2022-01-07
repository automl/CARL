import copy

from .envolvable_mlp import EvolvableMLP


class Individual():

    def __init__(self, state_dim, action_dim, actor_config, critic_config, rl_config, index, td3_double_q,
                 critic_2_config=None, replay_memory=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.rl_config = rl_config
        self.index = index
        self.td3_double_q = td3_double_q

        if critic_2_config is None:
            critic_2_config = copy.deepcopy(critic_config)

        self.actor = EvolvableMLP(num_inputs=state_dim, num_outputs=action_dim, **actor_config)
        self.critic_1 = EvolvableMLP(num_inputs=state_dim + action_dim, num_outputs=1, **critic_config)
        if td3_double_q:
            self.critic_2 = EvolvableMLP(num_inputs=state_dim + action_dim, num_outputs=1, **critic_2_config)

        self.fitness = []
        self.improvement = 0
        self.train_log = {"pre_fitness": None, "pre_rank": None, "post_fitness": None, "post_rank": None, "eval_eps": 0,
                          "index": None, "parent_index": None, "mutation": None}

        self.replay_memory = replay_memory

    def clone(self, index=None):
        if index is None:
            index = self.index

        if self.td3_double_q:
            critic_2_config = copy.deepcopy(self.critic_2.short_dict)
        else:
            critic_2_config = None

        clone = type(self)(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           actor_config=copy.deepcopy(self.actor.short_dict),
                           critic_config=copy.deepcopy(self.critic_1.short_dict),
                           rl_config=copy.deepcopy(self.rl_config),
                           index=index,
                           td3_double_q=self.td3_double_q,
                           critic_2_config=critic_2_config,
                           replay_memory=self.replay_memory)

        clone.fitness = copy.deepcopy(self.fitness)
        clone.train_log = copy.deepcopy(self.train_log)
        clone.actor = self.actor.clone()
        clone.critic_1 = self.critic_1.clone()
        if self.td3_double_q:
            clone.critic_2 = self.critic_2.clone()

        if self.replay_memory:
            self.replay_memory = copy.deepcopy(self.replay_memory)

        return clone
