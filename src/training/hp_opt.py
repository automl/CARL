import logging
logging.basicConfig(level=logging.INFO)
import os
from pathlib import Path

import sys
sys.path.append("..")
sys.path.append("../..")

print(os.system('python --version'))

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

from smac.configspace import ConfigurationSpace, Configuration
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

from src.train import get_parser, main
from src.training.trial_logger import TrialLogger


def carl_from_cfg(cfg, seed, budget, parser, args):
    """
    Creates a MLP classifier from sklearn and fits the given data on it.

    Parameters
    ----------
    cfg: Configuration
        configuration chosen by smac
    seed: int or RandomState
        seeeeed
    budget: float
        used to set max steps for the agent

    Returns
    -------
    float
    """
    opt_hyperparams = cfg
    args.budget = budget
    # args.seed = seed
    unknown_args = []

    # in CARL we try to maximize the reward
    final_ep_mean_reward = main(args=args, parser=parser, unknown_args=unknown_args, opt_hyperparams=opt_hyperparams)

    return -final_ep_mean_reward


def test_carl_from_cfg(carl_from_cfg_helper, cs):
    values = {
        "ent_coef": 0.3639574736567951,
        "gae_lambda": 0.9920997843658639,
        "gamma": 0.9066381701127927,
        "learning_rate": 5.043346488136671e-05,
        "max_grad_norm": 0.9484371657462239,
        "vf_coef": 0.1854654290187061,
    }
    configuration = Configuration(
        configuration_space=cs,
        values=values,
    )

    carl_from_cfg_helper(configuration, 123, 2000)


if __name__ == '__main__':
    cwd = os.getcwd()
    if "training" in cwd:
        os.chdir(str(Path(cwd).parent))
    print(os.getcwd())
    limit_mem_mb = 1e5
    t_limit_ta = 0.01 * 3600  # runtime limit for target algorithm [seconds]
    wallclock_limit_s = 0.5 * 1 * 3600

    parser = get_parser()

    args, unknown_args = parser.parse_known_args()

    logger = TrialLogger(
        args.outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=args.add_context_feature_names_to_logdir,
        init_sb3_tensorboard=False  # set to False if using SubprocVecEnv
    )

    smac_outdir = Path(logger.logdir) / "smac_logs"
    del logger

    budget_init = int(5000)  # TODO find values for environment families
    budget_max = int(args.steps)

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
        # 'wallclock-limit': wallclock_limit_s,  # max duration to run the optimization (in seconds)
        'cs': cs,  # configuration space
        'deterministic': 'true',
        'limit_resources': True,  # Uses pynisher to limit memory and runtime
                                  # Alternatively, you can also disable this.
                                  # Then you should handle runtime and memory yourself in the TA
        # 'cutoff': t_limit_ta,  # runtime limit for target algorithm
        'memory_limit': limit_mem_mb,  # adapt this to reasonable value for your hardware
        'output_dir': smac_outdir,
        'save_instantly': True,
    })

    # Intensifier parameters
    intensifier_kwargs = {'initial_budget': budget_init, 'max_budget': budget_max, 'eta': 3}

    # Initial design parameters
    initial_design_kwargs = {  # static
        "n_configs_x_params": 2,
        "max_config_fracs": 0.05,
        "init_budget": 1,
    }

    # To optimize, we pass the function to the SMAC-object
    carl_from_cfg_helper = partial(
        carl_from_cfg,
        parser=parser,
        args=args
    )
    
    # test_carl_from_cfg(carl_from_cfg_helper, cs)

    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(args.seed),
        tae_runner=carl_from_cfg_helper,
        # tae_runner_kwargs={'num_workers': 4},
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

    # inc_value = smac.get_tae_runner().run(
    #     config=incumbent,
    #     budget=budget_max,
    #     seed=0)[1]
    #
    # print('Optimized Value: %.4f' % inc_value)

    print(incumbent)
