import numpy as np
from functools import partial

import hydra
from omegaconf import DictConfig

from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace, Configuration
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI


def context_features_to_configuration_space(env_name: str) -> ConfigurationSpace:
    return None


# def eval_agent(config_smac: Configuration, cfg: DictConfig) -> float:
#     # Instantiate agent
#
#     # Instantiate env
#
#     for i in range(cfg.n_eval_episodes):
#         # Rollout
#         pass
#
#     cumulative_reward = 0.
#     # TODO normalize reward?
#     return cumulative_reward
#
#
# def create_tae_runner(cfg: DictConfig) -> callable:
#     return partial(
#         eval_agent,
#         cfg=cfg
#     )
#
#
# @hydra.main("./configs", "base")
# def main(cfg: DictConfig):
#
#     configuration_space = context_features_to_configuration_space(env_name=...)
#     scenario = Scenario({
#             "run_obj": "quality",  # we optimize quality (alternatively runtime)
#             "runcount-limit": cfg.budget,  # max. number of function evaluations
#             "cs": configuration_space,  # configuration space
#             "deterministic": True,
#         }
#     )
#
#     tae_runner = create_tae_runner(cfg=cfg)
#     model_type = "gp"
#     smac = SMAC4BB(
#         scenario=scenario,
#         model_type=model_type,
#         rng=np.random.RandomState(cfg.seed),
#         acquisition_function=EI,  # or others like PI, LCB as acquisition functions
#         tae_runner=tae_runner,
#     )
#
#     smac.optimize()


if __name__ == "__main__":
    # main()
    cs = context_features_to_configuration_space(env_name="CARLPendulum")
    print(cs)
