import numpy as np
import torch
import copy

import sys
sys.path.append("..")
from src.training.hpo.SEARL.searl.neuroevolution.components.replay_memory import MPReplayMemory, ReplayMemory
from src.training.hpo.SEARL.searl.neuroevolution.evaluation_td3 import MPEvaluation
from src.training.hpo.SEARL.searl.neuroevolution.mutation_mlp import Mutations
from src.training.hpo.SEARL.searl.neuroevolution.tournament_selection import TournamentSelection
from src.training.hpo.SEARL.searl.neuroevolution.training_td3 import TD3Training
from src.training.hpo.SEARL.searl.neuroevolution.components.utils import Transition
from src.training.hpo.SEARL.searl.neuroevolution.searl_td3 import SEARLforTD3
from src.training.hpo.SEARL.searl.neuroevolution.components.envolvable_mlp import EvolvableMLP
from src.training.hpo.SEARL.searl.neuroevolution.components.individual_td3 import Individual
from src.training.hpo.SEARL.searl.neuroevolution.components.utils import to_tensor, Transition

from src.training.hpo.make_searl_env import make_searl_env


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("train CUDA", device == torch.device("cuda"), device)


class CustomMPEvaluation(MPEvaluation):
    @staticmethod
    def _evaluate_individual(individual, config, num_episodes, seed, exploration_noise=False, start_phase=False):
        actor_net = individual.actor

        num_frames = 0
        fitness_list = []
        transistions_list = []
        episodes = 0

        env = make_searl_env(env_name=config.env.name)
        env.seed(seed)
        actor_net.eval()

        with torch.no_grad():
            while episodes < num_episodes or num_frames < config.eval.min_eval_steps:
                episode_fitness = 0.0
                episode_transitions = []
                state = env.reset()
                t_state = to_tensor(state).unsqueeze(0)
                done = False
                while not done:
                    if start_phase:
                        action = env.action_space.sample()
                        action = to_tensor(action)
                    else:
                        action = actor_net(t_state)
                    action.clamp(-1, 1)
                    action = action.data.numpy()
                    if exploration_noise is not False:
                        action += config.eval.exploration_noise * np.random.randn(config.action_dim)
                        action = np.clip(action, -1, 1)
                    action = action.flatten()

                    step_action = (action + 1) / 2  # [-1, 1] => [0, 1]
                    step_action *= (env.action_space.high - env.action_space.low)
                    step_action += env.action_space.low

                    next_state, reward, done, info = env.step(step_action)  # Simulate one step in environment

                    done_bool = 0 if num_frames + 1 == env._max_episode_steps else float(done)

                    t_next_state = to_tensor(next_state).unsqueeze(0)

                    episode_fitness += reward
                    num_frames += 1

                    transition = Transition(state, action, next_state, np.array([reward]),
                                            np.array([done_bool]).astype('uint8'))
                    episode_transitions.append(transition)
                    t_state = t_next_state
                    state = next_state
                episodes += 1
                fitness_list.append(episode_fitness)
                transistions_list.append(episode_transitions)

        return {
            individual.index: {"fitness_list": fitness_list, "num_episodes": num_episodes, "num_frames": num_frames,
                               "id": individual.index, "transitions": transistions_list}}



class CustomEvolvableMLP(EvolvableMLP):
    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0)

        state = state.to(self.device)

        dist = self.forward(state).data.cpu()
        action = dist.numpy()[0]
        return action


class CustomIndividual(Individual):
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

        self.actor = CustomEvolvableMLP(num_inputs=state_dim, num_outputs=action_dim, **actor_config)
        self.critic_1 = CustomEvolvableMLP(num_inputs=state_dim + action_dim, num_outputs=1, **critic_config)
        if td3_double_q:
            self.critic_2 = CustomEvolvableMLP(num_inputs=state_dim + action_dim, num_outputs=1, **critic_2_config)

        self.fitness = []
        self.improvement = 0
        self.train_log = {"pre_fitness": None, "pre_rank": None, "post_fitness": None, "post_rank": None,
                          "eval_eps": 0,
                          "index": None, "parent_index": None, "mutation": None}

        self.replay_memory = replay_memory


class CustomSEARLforTD3(SEARLforTD3):
    def __init__(self, config, logger, checkpoint):

        self.cfg = config
        self.log = logger
        self.ckp = checkpoint

        torch.manual_seed(self.cfg.seed.torch)
        np.random.seed(self.cfg.seed.numpy)

        self.log.print_config(self.cfg)
        self.log.csv.fieldnames(
            ["epoch", "time_string", "eval_eps", "pre_fitness", "pre_rank", "post_fitness", "post_rank", "index",
             "parent_index", "mutation", "train_iterations",
             ] + list(self.cfg.rl.get_dict.keys()))

        self.log.log("initialize replay memory")
        if self.cfg.nevo.ind_memory:
            push_queue = None
            sample_queue = None
        else:
            self.replay_memory = MPReplayMemory(seed=self.cfg.seed.replay_memory,
                                                capacity=self.cfg.train.replay_memory_size,
                                                batch_size=self.cfg.rl.batch_size,
                                                reuse_batch=self.cfg.nevo.reuse_batch)
            push_queue = self.replay_memory.get_push_queue()
            sample_queue = self.replay_memory.get_sample_queue()

        self.eval = CustomMPEvaluation(config=self.cfg, logger=self.log, push_queue=push_queue)

        self.tournament = TournamentSelection(config=self.cfg)

        self.mutation = Mutations(config=self.cfg, replay_sample_queue=sample_queue)

        self.training = TD3Training(config=self.cfg, replay_sample_queue=sample_queue)

    def initial_population(self):
        self.log.log("initialize population")
        population = []
        for idx in range(self.cfg.nevo.population_size):

            if self.cfg.nevo.ind_memory:
                replay_memory = ReplayMemory(capacity=self.cfg.train.replay_memory_size,
                                             batch_size=self.cfg.rl.batch_size)
            else:
                replay_memory = False

            if self.cfg.nevo.init_random:

                min_lr = 0.00001
                max_lr = 0.005

                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.rl)

                actor_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]
                critic_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]

                lr_actor = np.exp(np.random.uniform(np.log(min_lr), np.log(max_lr), 1))[0]
                lr_critic = np.exp(np.random.uniform(np.log(min_lr), np.log(max_lr), 1))[0]

                rl_config.set_attr("lr_actor", lr_actor)
                rl_config.set_attr("lr_critic", lr_critic)
                self.log(f"init {idx} rl_config: ", rl_config.get_dict)
                self.log(f"init {idx} actor_config: ", actor_config)

            else:
                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.rl)

            indi = CustomIndividual(state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                              actor_config=actor_config,
                              critic_config=critic_config,
                              rl_config=rl_config, index=idx, td3_double_q=self.cfg.train.td3_double_q,
                              replay_memory=replay_memory)
            population.append(indi)
        return population