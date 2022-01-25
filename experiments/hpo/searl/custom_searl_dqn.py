import numpy as np
import torch
from experiments.hpo.searl.SEARL.searl.neuroevolution.components.replay_memory import ReplayMemory
from experiments.hpo.searl.SEARL.searl.neuroevolution.mutation_cnn import Mutations
from experiments.hpo.searl.SEARL.searl.neuroevolution.searl_dqn import SEARLforDQN
from experiments.hpo.searl.SEARL.searl.neuroevolution.tournament_selection import TournamentSelection
from experiments.hpo.searl.SEARL.searl.neuroevolution.training_dqn import DQNTraining
from experiments.hpo.searl.SEARL.searl.neuroevolution.evaluation_dqn import MPEvaluation
from experiments.hpo.searl.SEARL.searl.neuroevolution.components.utils import Transition

from experiments.hpo.searl.make_searl_env import make_searl_env


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


class CustomSEARLforDQN(SEARLforDQN):
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
        self.eval = CustomMPEvaluation(config=self.cfg, logger=self.log, replay_memory=self.replay_memory)
        self.tournament = TournamentSelection(config=self.cfg)
        self.mutation = Mutations(config=self.cfg)
        self.training = DQNTraining(config=self.cfg, replay_memory=self.replay_memory)