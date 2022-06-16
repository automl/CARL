import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)
import numpy as np
from functools import partial
import importlib
from rich import print
from pathlib import Path
import time
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig
import coax

from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace, Configuration, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI

import carl.envs
from experiments.common.train.utils import make_carl_env
from experiments.common.utils.search_space_encoding import search_space_to_config_space
from experiments.attack_on_agents.agent_creation import make_agent


def evaluate(pi, env, num_episodes):
    returns = []
    transitions = []
    for i in range(num_episodes):
        ret = 0
        s = env.reset()

        for t in range(env.cutoff):
            a = pi.mean(s)  # use mean for exploitation  # TODO use mean?
            s_next, r, done, info = env.step(a)
            transition = (i, s, a, r, done)
            transitions.append(transition)
            ret += r
            if done:
                break
            s = s_next
        returns.append(ret)
    return returns, transitions


def context_features_to_configuration_space(env_name: str) -> ConfigurationSpace:
    env_cls = getattr(carl.envs, env_name)
    env_module = importlib.import_module(env_cls.__module__)
    context_def = getattr(env_module, "DEFAULT_CONTEXT")
    context_bounds = getattr(env_module, "CONTEXT_BOUNDS")
    print(context_bounds)
    hyperparameters = []
    for cf_name, cf_bounds in context_bounds.items():
        lb = cf_bounds[0]
        ub = cf_bounds[1]
        cf_type = cf_bounds[2]

        if cf_type == float:
            hp_cls = UniformFloatHyperparameter
        else:
            raise NotImplementedError

        hp = hp_cls(
            cf_name,
            lower=lb,
            upper=ub,
            default_value=context_def[cf_name],
            log=False,
            q=None,
            meta=None
        )
        hyperparameters.append(hp)
    configuration_space = ConfigurationSpace()
    configuration_space.add_hyperparameters(hyperparameters=hyperparameters)

    return configuration_space


def get_default_context(env_name: str) -> Dict[Any, Any]:
    env_cls = getattr(carl.envs, env_name)
    env_module = importlib.import_module(env_cls.__module__)
    context_def = getattr(env_module, "DEFAULT_CONTEXT")
    return context_def


def eval_agent(config_smac: Configuration, cfg: DictConfig, file_id: Optional[str] = None) -> float:
    # Instantiate env
    contexts = None
    context = None
    if config_smac is not None:
        context = {k: config_smac[k] for k in config_smac}
        contexts = {"0": context}
    eval_env = make_carl_env(cfg=cfg, contexts=contexts)

    # Instantiate agent
    policy = make_agent(cfg=cfg, env=eval_env)

    returns, transitions = evaluate(policy, eval_env, cfg.n_eval_episodes)

    mean_return = float(np.mean(returns))

    run_data = np.array({
        "context": context,
        "returns": returns,
        "transitions": transitions,
        "performance": mean_return
    })
    if file_id is None:
        file_id = time.time_ns()
    fp = Path(f"./eval_data/eval_data_{file_id}.npz")
    fp.parent.mkdir(exist_ok=True, parents=True)
    run_data.dump(fp)

    # TODO normalize reward?
    return mean_return


def create_tae_runner(cfg: DictConfig) -> callable:
    return partial(
        eval_agent,
        cfg=cfg
    )


@hydra.main("./configs", "base")
def main(cfg: DictConfig):
    print(cfg)
    configuration_space = search_space_to_config_space(search_space=cfg.context_feature_search_space)
    configuration_space.seed(cfg.seed)
    print(configuration_space)

    scenario = Scenario({
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": cfg.budget,  # max. number of function evaluations
            "cs": configuration_space,  # configuration space
            "deterministic": True,
        }
    )

    tae_runner = create_tae_runner(cfg=cfg)
    context_default = get_default_context(env_name=cfg.env)
    performance_default = tae_runner(context_default, file_id="default")

    model_type = "gp"
    smac = SMAC4BB(
        scenario=scenario,
        model_type=model_type,
        rng=np.random.RandomState(cfg.seed),
        acquisition_function=EI,  # or others like PI, LCB as acquisition functions
        tae_runner=tae_runner,
        initial_design_kwargs={"init_budget": 2}
    )

    smac.optimize()

    return configuration_space


if __name__ == "__main__":
    main()
    # cs = context_features_to_configuration_space(env_name="CARLPendulumEnv")
