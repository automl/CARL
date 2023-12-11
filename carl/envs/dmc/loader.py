from typing import Any, Dict, Optional

import inspect

import dm_env  # type: ignore
from dm_control import suite  # type: ignore

from carl.envs.dmc.dmc_tasks import (  # type: ignore [import]  # noqa: F401
    finger,
    fish,
    pointmass,
    quadruped,
    walker,
)
from carl.utils.types import Context

_DOMAINS = {
    name: module
    for name, module in locals().items()
    if inspect.ismodule(module) and hasattr(module, "SUITE")
}


def load_dmc_env(
    domain_name: str,
    task_name: str,
    context: Context = {},
    task_kwargs: Optional[Any] = None,
    environment_kwargs: Dict[str, bool] = None,
    visualize_reward: bool = False,
) -> dm_env:
    if domain_name in _DOMAINS:
        domain = _DOMAINS[domain_name]
    elif domain_name in suite._DOMAINS:
        domain = suite._DOMAINS[domain_name]
    else:
        raise ValueError("Domain {!r} does not exist.".format(domain_name))

    if task_name in domain.SUITE:
        task_kwargs = task_kwargs or {}
        if environment_kwargs is not None:
            task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
        env = domain.SUITE[task_name](context=context, **task_kwargs)
        env.task.visualize_reward = visualize_reward
        return env
    elif (domain_name, task_name) in suite.ALL_TASKS:
        return suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    else:
        raise ValueError(
            "Task {!r} does not exist in domain {!r}.".format(task_name, domain_name)
        )
