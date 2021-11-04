import logging
logging.basicConfig(level=logging.INFO)

import warnings
import numpy as np
from functools import partial

import ConfigSpace as CS
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

from src.train import get_parser, main

# Target Algorithm
def carl_from_cfg(cfg, seed, budget, args):
    """
    Creates a MLP classifier from sklearn and fits the given data on it.

    Parameters
    ----------
    cfg: Configuration
        configuration chosen by smac
    seed: int or RandomState
        used to initialize the rf's random generator
    budget: float
        used to set max iterations for the MLP

    Returns
    -------
    float
    """


    opt_hyperparams = cfg
    args.budget = budget
    args.seed = seed

    # in CARL we try to maximize the reward
    final_ep_mean_reward = main(args=args, unknown_args=unknown_args, opt_hyperparams=opt_hyperparams)

    return -final_ep_mean_reward


if __name__ == '__main__':
    limit_mem_mb = 1e5
    t_limit_ta = 3 * 3600  # runtime limit for target algorithm [seconds]
    wallclock_limit_s = 2*24*3600

    parser = get_parser()

    budget_init = 1000  # TODO find values for environment families

    args, unknown_args = parser.parse_known_args()

    budget_max = args.steps

    # main(args, unknown_args, parser)
    # DDPG: learning rate, gamma, tau

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    if args.agent == "DDPG":
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', 0.00001, 0.02, default_value=3e-5, log=True
        )
        gamma = UniformFloatHyperparameter(
            'gamma', 0.8, 0.999, default_value=0.95, log=True
        )
        tau = UniformFloatHyperparameter(
            'tau', 1e-3, 0.99, default_value=0.005, log=True
        )
        cs.add_hyperparameters([
            learning_rate,
            gamma,
            tau
            ])
    elif args.agent == "PPO":
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', 0.00001, 0.02, default_value=3e-5, log=True
        )
        gamma = UniformFloatHyperparameter(
            'gamma', 0.8, 0.999, default_value=0.99, log=True
        )
        gae_lambda = UniformFloatHyperparameter(
            'gae_lambda', 0.8, 0.999, default_value=0.99, log=True
        )
        ent_coef = UniformFloatHyperparameter(
            'ent_coef', 0., 0.5, default_value=0.
        )
        max_grad_norm = UniformFloatHyperparameter(
            'max_grad_norm', 0., 1., default_value=0.5
        )
        vf_coef = UniformFloatHyperparameter(
            'vf_coef', 0., 1., default_value=0.5
        )
        cs.add_hyperparameters([
            learning_rate,
            gamma,
            gae_lambda,
            ent_coef,
            max_grad_norm,
            vf_coef
            ])
    else:
        raise ValueError(f"Agent {args.agent} not supported.")

    # SMAC scenario object
    scenario = Scenario({
        'run_obj': 'quality',  # we optimize quality (alternative to runtime)
        'wallclock-limit': wallclock_limit_s,  # max duration to run the optimization (in seconds)
        'cs': cs,  # configuration space
        'deterministic': 'true',
        'limit_resources': True,  # Uses pynisher to limit memory and runtime
                                  # Alternatively, you can also disable this.
                                  # Then you should handle runtime and memory yourself in the TA
        'cutoff': t_limit_ta,  # runtime limit for target algorithm
        'memory_limit': limit_mem_mb,  # adapt this to reasonable value for your hardware
    })

    # Intensifier parameters
    intensifier_kwargs = {'initial_budget': budget_init, 'max_budget': budget_max, 'eta': 3}

    # To optimize, we pass the function to the SMAC-object
    carl_from_cfg_helper = partial(
        carl_from_cfg,
        args=args
    )

    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(args.seed),
        tae_runner=carl_from_cfg_helper,
        intensifier_kwargs=intensifier_kwargs,
    )

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    # def_value = smac.get_tae_runner().run(
    #     config=cs.get_default_configuration(),
    #     budget=budget_max,
    #     seed=0)[1]
    #
    # print('Value for default configuration: %.4f' % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(
        config=incumbent,
        budget=budget_max,
        seed=0)[1]

    print('Optimized Value: %.4f' % inc_value)
