from typing import List

import numpy as np
import torch

from .components.utils import Transition
from ..rl_algorithms.components.wrappers import make_atari, wrap_deepmind, wrap_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("train CUDA", device == torch.device("cuda"), device)


class MPEvaluation():
    def __init__(self, config, logger, replay_memory=None):

        self.rng = np.random.RandomState(config.seed.evaluation)
        self.cfg = config
        self.log = logger
        self.push_queue = replay_memory
        self.eval_episodes = config.eval.eval_episodes

    def test_individual(self, individual, epoch):
        return_dict = self._evaluate_individual(individual, self.cfg, self.cfg.eval.test_episodes, epoch, False)
        fitness = np.mean(return_dict[individual.index]["fitness_list"])
        return fitness

    @staticmethod
    def _evaluate_individual(individual, config, num_episodes, seed, exploration_noise=False, start_phase=False):

        actor_net = individual.actor

        num_frames = 0
        fitness_list = []
        transistions_list = []
        episodes = 0

        env = make_atari(config.env.name)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
        env.seed(seed)

        actor_net.eval()
        actor_net.to(device)
        actor_net.device = device

        with torch.no_grad():
            while episodes < num_episodes or num_frames < config.eval.min_eval_steps:
                episode_fitness = 0.0
                episode_transitions = []
                state = env.reset()

                done = False
                while not done:
                    action = actor_net.act(state)

                    next_state, reward, done, info = env.step(action)
                    episode_fitness += reward
                    num_frames += 1

                    transition = Transition(torch.FloatTensor(state), torch.LongTensor([action]),
                                            torch.FloatTensor(next_state), torch.FloatTensor(np.array([reward])),
                                            torch.FloatTensor(np.array([done]).astype('uint8'))
                                            )

                    episode_transitions.append(transition)
                    state = next_state
                episodes += 1
                fitness_list.append(episode_fitness)
                transistions_list.append(episode_transitions)

        actor_net.to(torch.device("cpu"))

        return {individual.index: {"fitness_list": fitness_list, "num_episodes": num_episodes, "num_frames": num_frames,
                                   "id": individual.index, "transitions": transistions_list}}

    def evaluate_population(self, population: List, exploration_noise=False, total_frames=1):

        population_id_lookup = [ind.index for ind in population]
        new_population_mean_fitness = np.zeros(len(population))
        new_population_var_fitness = np.zeros(len(population))

        start_phase = total_frames <= self.cfg.rl.start_timesteps
        if start_phase:
            self.log("start phase", time_step=total_frames)

        args_list = [(ind, self.cfg, self.eval_episodes, self.rng.randint(0, 100000), exploration_noise, start_phase)
                     for ind in population]

        result_dict = []
        for args in args_list:
            result_dict.append(self._evaluate_individual(*args))

        eval_frames = 0
        for list_element in result_dict:
            for ind_id, value_dict in list_element.items():
                pop_idx = population_id_lookup.index(ind_id)
                new_population_mean_fitness[pop_idx] = np.mean(value_dict['fitness_list'])
                new_population_var_fitness[pop_idx] = np.var(value_dict['fitness_list'])
                eval_frames += value_dict['num_frames']

                population[pop_idx].train_log["eval_eps"] = self.eval_episodes

                for transitions in value_dict['transitions']:
                    if self.cfg.nevo.ind_memory:
                        population[pop_idx].replay_memory.add(transitions)
                    else:
                        self.push_queue.put(transitions)

        for idx in range(len(population)):
            population[idx].train_log["post_fitness"] = new_population_mean_fitness[idx]
            population[idx].train_log["index"] = population[idx].index
            self.log.csv.log_csv(population[idx].train_log)
            population[idx].train_log.update(
                {"pre_fitness": new_population_mean_fitness[idx],
                 "eval_eps": 0})  # , "pre_rank": population_rank[idx], "eval_eps":0}
            population[idx].fitness.append(new_population_mean_fitness[idx])
            if len(population[idx].fitness) > 1:
                population[idx].improvement = population[idx].fitness[-1] - population[idx].fitness[-2]
            else:
                population[idx].improvement = population[idx].fitness[-1]

        return new_population_mean_fitness, new_population_var_fitness, eval_frames
