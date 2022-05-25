from dm_control import suite

from carl.envs.dmc.carl_dmcontrol import CARLDmc


def load_dmc_env(domain_name, task_name, task_kwargs=None, environment_kwargs=None,
                 visualize_reward=False):
    return suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
    )


def load_dmc_cartpole():
    return load_dmc_env(domain_name="cartpole", task_name="swingup")

# TODO Find a good method how to define tasks. Define classes? Better, create an automatic class constructor


if __name__ == "__main__":
    # Load one task:
    env = load_dmc_cartpole()
    carl_env = CARLDmc(env=env)

    s = carl_env.reset()
    done = False
    while not done:
        action = carl_env.action_space.sample()
        state, reward, done, info = carl_env.step(action=action)
        print(reward, done)

    # # Iterate over a task set:
    # for domain_name, task_name in suite.BENCHMARKING:
    #     env = suite.load(domain_name, task_name)
    #
    # # Step through an episode and print out reward, discount and observation.
    # action_spec = env.action_spec()
    # time_step = env.reset()
    # while not time_step.last():
    #     action = np.random.uniform(
    #         action_spec.minimum, action_spec.maximum, size=action_spec.shape
    #     )
    #     time_step = env.step(action)
    #     print(time_step.reward, time_step.discount, time_step.observation)
