from typing import Union, Type
import sys
sys.path.append("../..")
from src.training.hpo.searl.SEARL.searl.utils.supporter import Supporter
from src.training.hpo.searl.make_searl_env import make_searl_env
from src.training.hpo.searl.custom_searl_td3 import CustomSEARLforTD3
from src.training.hpo.searl.custom_searl_dqn import CustomSEARLforDQN


def start_searl_run(
        config_dict, expt_dir, searl_algorithm: Type[Union[CustomSEARLforTD3, CustomSEARLforDQN]] = CustomSEARLforDQN):
    sup = Supporter(experiments_dir=expt_dir, config_dict=config_dict, count_expt=True)
    cfg = sup.get_config()
    log = sup.get_logger()

    env = make_searl_env(env_name=cfg.env.name)
    cfg.set_attr("action_dim", env.action_space.shape[0])
    cfg.set_attr("state_dim", env.observation_space.shape[0])

    # TODO set num_episodes for eval
    # TODO save one context set and reuse. Now: for every evaluation env a new set of contexts is sampled from the same distribution

    searl = searl_algorithm(config=cfg, logger=log, checkpoint=sup.ckp)

    population = searl.initial_population()
    searl.evolve_population(population)
