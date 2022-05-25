# flake8: noqa: W605
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from carl import envs


def get_default_context_and_bounds(
    env_name: str,
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    Get context feature defaults and bounds for environment.

    Parameters
    ----------
    env_name: str
        Name of CARLEnv.

    Returns
    -------
    Tuple[Dict[Any, Any], Dict[Any, Any]]
        Context feature defaults as dictionary, context feature bounds as dictionary.
        Keys are the names of the context features.

        Context feature bounds can be in following formats:
            int/float context features:
                ``"MAIN_ENGINE_POWER": (0, 50, float)``

            list of int/float context feature:
                ``"target_structure_ids": (0, np.inf, [list, int])``

            categorical context features:
                ``"VEHICLE": (None, None, "categorical", np.arange(0, len(PARKING_GARAGE)))``
    """
    # TODO make less hacky / make explicit
    env_defaults = getattr(envs, f"{env_name}_defaults")
    env_bounds = getattr(envs, f"{env_name}_bounds")

    return env_defaults, env_bounds


def sample_contexts(
    env_name: str,
    context_feature_args: List[str],
    num_contexts: int,
    default_sample_std_percentage: float = 0.05,
    fallback_sample_std: float = 0.1,
) -> Dict[int, Dict[str, Any]]:
    """
    Sample contexts.

    Control which/how the context features are sampled with `context_feature_args`.
    Categorical context features are sampled radonmly via the given choices in the context bounds.
    For continuous context features a new value is sampled in the following way:

    .. math:: x_{cf,new} \sim \mathcal{N}(x_{cf, default}, \sigma_{rel} \cdot x_{cf, default})

    :math:`x_{cf,new}`: New context feature value

    :math:`x_{cf, default}`: Default context feature value

    :math:`\sigma_{rel}`: Relative standard deviation, parametrized in  `context_feature_args`
        by providing e.g. `["<context_feature_name>_std", "0.05"]`.

    Examples
    --------
    Sampling two contexts for the CARLAcrobotEnv and changing only the context feature link_length_2.
    >>> sample_contexts("CARLAcrobotEnv", ["link_length_2"], 2)
    {0: {'link_length_1': 1,
      'link_length_2': 1.0645201049835367,
      'link_mass_1': 1,
      'link_mass_2': 1,
      'link_com_1': 0.5,
      'link_com_2': 0.5,
      'link_moi': 1,
      'max_velocity_1': 12.566370614359172,
      'max_velocity_2': 28.274333882308138},
     1: {'link_length_1': 1,
      'link_length_2': 1.011885635790618,
      'link_mass_1': 1,
      'link_mass_2': 1,
      'link_com_1': 0.5,
      'link_com_2': 0.5,
      'link_moi': 1,
      'max_velocity_1': 12.566370614359172,
      'max_velocity_2': 28.274333882308138}}


    Parameters
    ----------
    env_name: str
        Name of MetaEnvironment
    context_feature_args: List[str]
        All arguments from the parser, e.g., ["context_feature_0", "context_feature_1", "context_feature_1_std", "0.05"]
    num_contexts: int
        Number of contexts to sample.
    default_sample_std_percentage: float, optional
        The default relative standard deviation to use if <context_feature_name>_std is not specified. The default is
        0.05.
    fallback_sample_std: float, optional
        The fallback relative standard deviation. Defaults to 0.1.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Dictionary containing the sampled contexts. Keys are integers, values are Dicts containing the context feature
        names as keys and context feature values as values, e.g.,

    """
    # Get default context features and bounds
    env_defaults, env_bounds = get_default_context_and_bounds(env_name=env_name)

    # Create sample distributions/rules
    sample_dists = {}
    for context_feature_name in env_defaults.keys():
        if context_feature_name in context_feature_args:
            if f"{context_feature_name}_mean" in context_feature_args:
                sample_mean = float(
                    context_feature_args[
                        context_feature_args.index(f"{context_feature_name}_mean") + 1
                    ]
                )
            else:
                sample_mean = env_defaults[context_feature_name]

            if f"{context_feature_name}_std" in context_feature_args:
                sample_std = float(
                    context_feature_args[
                        context_feature_args.index(f"{context_feature_name}_std") + 1
                    ]
                )
            else:
                sample_std = default_sample_std_percentage * np.abs(sample_mean)

            if sample_mean == 0:
                # Fallback sample standard deviation. Necessary if the sample mean is 0.
                # In this case the sample standard deviation would be 0 as well and we would always sample
                # the sample mean. Therefore we use a fallback sample standard deviation.
                sample_std = fallback_sample_std  # TODO change this back to sample_std

            random_variable = norm(loc=sample_mean, scale=sample_std)
            context_feature_type = env_bounds[context_feature_name][2]
            sample_dists[context_feature_name] = (random_variable, context_feature_type)

    # Sample contexts
    contexts = {}
    for i in range(0, num_contexts):
        c = {}
        # k = name of context feature
        for k in env_defaults.keys():
            if k in sample_dists.keys():
                # If we have a special sampling distribution/rule for context feature k
                random_variable = sample_dists[k][0]
                context_feature_type = sample_dists[k][1]
                lower_bound, upper_bound = env_bounds[k][0], env_bounds[k][1]
                if context_feature_type == list:
                    length = np.random.randint(
                        5e5
                    )  # TODO should we allow lists to be this long? or should we parametrize this?
                    arg_class = sample_dists[k][1][1]
                    context_list = random_variable.rvs(size=length)
                    context_list = np.clip(context_list, lower_bound, upper_bound)
                    c[k] = [arg_class(c) for c in context_list]
                elif context_feature_type == "categorical":
                    choices = env_bounds[k][3]
                    choice = np.random.choice(choices)
                    c[k] = choice
                elif context_feature_type == "conditional":
                    condition = env_bounds[k][4]
                    choices = env_bounds[k][3][condition]
                    choice = np.random.choice(choices)
                    c[k] = choice
                else:
                    c[k] = random_variable.rvs(size=1)[0]  # sample variable
                    c[k] = np.clip(c[k], lower_bound, upper_bound)  # check bounds
                    c[k] = context_feature_type(c[k])  # cast to given type
            else:
                # No special sampling rule for context feature k, use the default context feature value
                c[k] = env_defaults[k]
        contexts[i] = c

    return contexts
