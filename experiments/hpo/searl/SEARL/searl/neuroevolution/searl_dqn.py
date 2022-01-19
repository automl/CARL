import copy
import time

import numpy as np
import torch

from .components.individual_dqn import DQNIndividual
from .components.replay_memory import ReplayMemory
from .evaluation_dqn import MPEvaluation
from .mutation_cnn import Mutations
from .tournament_selection import TournamentSelection
from .training_dqn import DQNTraining
from ..rl_algorithms.components.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from ..utils.supporter import Supporter


class SEARLforDQN():

    def __init__(self, config, logger, checkpoint):

        self.cfg = config
        self.log = logger
        self.ckp = checkpoint

        torch.manual_seed(self.cfg.seed.torch)
        np.random.seed(self.cfg.seed.numpy)

        self.log.print_config(self.cfg)
        self.log.csv.fieldnames(
            ["epoch", "time_string", "eval_eps", "pre_fitness", "pre_rank", "post_fitness", "post_rank", "index",
             "parent_index", "mutation", "train_iterations", "train_losses",
             ] + list(self.cfg.rl.get_dict.keys()))

        self.log.log("initialize replay memory")

        self.replay_memory = ReplayMemory(capacity=self.cfg.train.replay_memory_size, batch_size=self.cfg.rl.batch_size)

        self.eval = MPEvaluation(config=self.cfg, logger=self.log, replay_memory=self.replay_memory)

        self.tournament = TournamentSelection(config=self.cfg)

        self.mutation = Mutations(config=self.cfg)

        self.training = DQNTraining(config=self.cfg, replay_memory=self.replay_memory)

    def initial_population(self):
        self.log.log("initialize population")
        population = []
        for idx in range(self.cfg.nevo.population_size):

            if self.cfg.nevo.ind_memory:
                replay_memory = ReplayMemory(capacity=self.cfg.train.replay_memory_size,
                                             batch_size=self.cfg.rl.batch_size)
            else:
                replay_memory = False

            actor_config = copy.deepcopy(self.cfg.actor.get_dict)
            rl_config = copy.deepcopy(self.cfg.rl)

            indi = DQNIndividual(state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                                 actor_config=actor_config,
                                 rl_config=rl_config, index=idx, replay_memory=replay_memory)
            population.append(indi)
        return population

    def evolve_population(self, population, epoch=1, num_frames=0):

        frames_since_mut = 0
        num_frames = num_frames
        epoch = epoch

        while True:
            epoch_time = time.time()
            self.log(f"##### START EPOCH {epoch}", time_step=num_frames)

            for ind in population:
                ind.train_log['epoch'] = epoch

            population_mean_fitness, population_var_fitness, eval_frames = \
                self.log.log_func(self.eval.evaluate_population, population=population,
                                  exploration_noise=self.cfg.eval.exploration_noise,
                                  total_frames=num_frames)
            self.log("eval_frames", eval_frames)
            num_frames += eval_frames
            frames_since_mut += eval_frames

            self.log.population_info(population_mean_fitness, population_var_fitness, population, num_frames, epoch)

            self.ckp.save_object(population, name="population")
            self.log.log("save population")

            if num_frames >= self.cfg.train.num_frames:
                break

            if self.cfg.nevo.selection:
                elite, population = self.log.log_func(self.tournament.select, population)
                test_fitness = self.eval.test_individual(elite, epoch)
                self.log(f"##### ELITE INFO {epoch}", time_step=num_frames)
                self.log("best_test_fitness", test_fitness, num_frames)

            if self.cfg.nevo.mutation:
                population = self.log.log_func(self.mutation.mutation, population)

            if self.cfg.nevo.training:
                iterations = min(
                    max(self.cfg.train.min_train_steps, int(self.cfg.rl.train_frames_fraction * eval_frames)),
                    self.cfg.train.max_train_steps)
                self.log("training_iterations", iterations)
                population = self.log.log_func(self.training.train, population=population, iterations=iterations)

            self.log(f"##### END EPOCH {epoch} - runtime {time.time() - epoch_time:6.1f}", time_step=num_frames)
            self.log("epoch", epoch, time_step=num_frames)
            self.log(f"##### ################################################# #####")
            self.cfg.expt.set_attr("epoch", epoch)
            self.cfg.expt.set_attr("num_frames", num_frames)
            epoch += 1

        self.log("FINISH", time_step=num_frames)
        self.replay_memory.close()

    def close(self):
        self.replay_memory.close()


def start_searl_dqn_run(config_dict, expt_dir):
    sup = Supporter(experiments_dir=expt_dir, config_dict=config_dict, count_expt=True)
    cfg = sup.get_config()
    log = sup.get_logger()

    env = make_atari(cfg.env.name)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    cfg.set_attr("action_dim", env.action_space.n)
    cfg.set_attr("state_dim", env.observation_space.shape)

    searl = SEARLforDQN(config=cfg, logger=log, checkpoint=sup.ckp)

    population = searl.initial_population()
    searl.evolve_population(population)
