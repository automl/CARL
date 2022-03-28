from stable_baselines3.common.type_aliases import Schedule
import numpy as np


class CosineAnnealingLRSchedule(object):
    def __init__(self, lr_min: float, lr_max: float):
        self.lr_min = lr_min
        self.lr_max = lr_max

    def __call__(self, remaining_progress: float) -> float:
        """
        Set the learning rate of each parameter group using a cosine annealing
        schedule, where :math:`\eta_{max}` is set to the initial lr and
        :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

        .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

         It has been proposed in
        `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
        implements the cosine annealing part of SGDR, and not the restarts.

        (Docstring copied from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR)

        .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

        Parameters
        ----------
        remaining_progress : float
            1 -> 0. 1 - T_cur / T_max

        Returns
        -------
        float
            Current learning rate

        """
        eta_min = self.lr_min
        eta_max = self.lr_max

        eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos((1 - remaining_progress) * np.pi))

        return eta
    
    
if __name__ == '__main__':
    T = np.linspace(1, 0, 100)
    step = np.arange(0, len(T))
    lr_fn = CosineAnnealingLRSchedule(1e-4, 1e-3)
    L = lr_fn(T)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.lineplot(x=step, y=L)
    ax.set_yscale("log")
    plt.show()



