import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(
        default_value: float, percentage_std: float = 0.01, random_generator: np.random.Generator = None):
    std = percentage_std * np.abs(default_value)
    mean = 0
    if not random_generator:
        random_generator = np.random.default_rng()
    value = default_value + random_generator.normal(loc=mean, scale=std)

    return value


if __name__ == "__main__":
    seed = 123456
    rng = np.random.default_rng(seed=seed)
    default_value = 10
    percentage_std = 0.01
    n_samples = 1000
    values = np.array([add_gaussian_noise(
        default_value=default_value, percentage_std=percentage_std, random_generator=rng) for i in range(n_samples)])
    plt.close('all')
    plt.hist(values, bins=100)
    plt.show()


