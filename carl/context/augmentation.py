import numpy as np
from typing import Union, List


def add_gaussian_noise(
    default_value: Union[float, List[float]],
    percentage_std: float = 0.01,
    random_generator: np.random.Generator = None,
) -> Union[float, List[float]]:
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 123456
    rng = np.random.default_rng(seed=seed)
    default_value = 10
    default_value = list(np.arange(0, 4))
    percentage_std = 0.01
    n_samples = 1000
    values = np.array(
        [
            add_gaussian_noise(
                default_value=default_value,
                percentage_std=percentage_std,
                random_generator=rng,
            )
            for i in range(n_samples)
        ]
    )
    plt.close("all")
    plt.hist(values, bins=100)
    plt.show()
