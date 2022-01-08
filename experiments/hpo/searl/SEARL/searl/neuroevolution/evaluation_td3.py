from collections import ChainMap
from typing import List

import gym
import numpy as np
import torch

from .components.utils import to_tensor, Transition


class MPEvaluation():
    """
    evaluates and population and stores transitions an a push_queue

    """

    def __init__(self, config, logger, push_queue=None):

        self.rng = np.random.RandomState(config.seed.evaluation)
        self.cfg = config
        self.log = logger
        self.push_queue = push_queue
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

        env = gym.make(config.env.name)
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

        return {individual.index: {"fitness_list": fitness_list, "num_episodes": num_episodes, "num_frames": num_frames,
                                   "id": individual.index, "transitions": transistions_list}}

    def evaluate_population(self, population: List, exploration_noise=False, total_frames=1, pool=None):
        population_id_lookup = [ind.index for ind in population]
        new_population_mean_fitness = np.zeros(len(population))
        new_population_var_fitness = np.zeros(len(population))

        start_phase = total_frames <= self.cfg.rl.start_timesteps
        if start_phase:
            self.log("start phase", time_step=total_frames)

        args_list = [(ind, self.cfg, self.eval_episodes, self.rng.randint(0, 100000), exploration_noise, start_phase)
                     for ind in population]
        result_dicts = [pool.apply(self._evaluate_individual, args) for args in args_list]
        result_dict = dict(ChainMap(*result_dicts))

        eval_frames = 0
        for ind_id, value_dict in result_dict.items():
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
            population[idx].train_log.update({"pre_fitness": new_population_mean_fitness[idx], "eval_eps": 0})
            population[idx].fitness.append(new_population_mean_fitness[idx])
            if len(population[idx].fitness) > 1:
                population[idx].improvement = population[idx].fitness[-1] - population[idx].fitness[-2]
            else:
                population[idx].improvement = population[idx].fitness[-1]

        return new_population_mean_fitness, new_population_var_fitness, eval_frames
