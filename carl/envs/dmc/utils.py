import inspect

from dm_control import suite

from carl.envs.dmc.dmc_tasks import cartpole

_DOMAINS = {name: module for name, module in locals().items() 
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}

def load_dmc_env(domain_name, task_name, context={}, task_kwargs=None, environment_kwargs=None,
                 visualize_reward=False):

    if domain_name not in _DOMAINS and domain_name not in suite._DOMAINS:
        raise ValueError('Domain {!r} does not exist.'.format(domain_name))

    domain = _DOMAINS[domain_name]

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
        raise ValueError('Level {!r} does not exist in domain {!r}.'.format(
            task_name, domain_name))