from typing import Any, List, Union

import numpy as np


def add_gaussian_noise(
    default_value: Union[float, List[float]],
    percentage_std: Union[float, Any] = 0.01,
    random_generator: np.random.Generator = None,
) -> Union[float, Any]:
    """
    Add gaussian noise to default value.

    Parameters
    ----------
    default_value: Union[float, List[float]]
        Mean of normal distribution. Can be a scalar or a list of floats. If it is a list(-like) with length n, the
        output will also be of length n.
    percentage_std: float, optional = 0.01
        Relative standard deviation, multiplied with default value (mean) is standard deviation of normal distribution.
        If the default value is 0, percentage_std is assumed to be the absolute standard deviation.
    random_generator: np.random.Generator, optional = None
        Optional random generator to ensure deterministic behavior.

    Returns
    -------
    Union[float, List[float]]
        Default value with gaussian noise. If input was list (or array) with length n, output is also list (or array)
        with length n.
    """
    if type(default_value) in [int, float] and default_value != 0:
        std = percentage_std * np.abs(default_value)
    else:
        std = percentage_std
    mean = np.zeros_like(default_value)
    if not random_generator:
        random_generator = np.random.default_rng()
    value = default_value + random_generator.normal(loc=mean, scale=std)

    return value
