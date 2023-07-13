from carl.context.context_space import NormalFloatContextFeature
from carl.context.sampler import ContextSampler
from carl.envs.gymnasium.classic_control.carl_cartpole import CARLCartPole


if __name__ == "__main__":
    from rich import print as printr

    seed = 0
    # Sampling demo
    context_distributions = [NormalFloatContextFeature("gravity", mu=9.8, sigma=1)]
    context_sampler = ContextSampler(
        context_distributions=context_distributions,
        context_space=CARLCartPole.get_context_space(),
        seed=seed,
    )
    contexts = context_sampler.sample_contexts(n_contexts=5)

    # Env demo

    printr(CARLCartPole.get_context_space())

    printr(contexts)


    obs_context_features = list(CARLCartPole.get_default_context().keys())[:2]

    env = CARLCartPole(contexts=contexts, obs_context_features=obs_context_features)
    env.render_mode = 'rgb_array'

    state = env.reset()

    print(env.context_id)
    print(env.context)
    env.context_id = 4
    print(env.context_id)
    print(env.context)

    printr(state)

    env.step(env.action_space.sample())
    print(env.render())

    env.reset()
    env.reset()
