if __name__ == "__main__":
    import functools
    import os
    from datetime import datetime

    import brax
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from brax import envs
    from brax.io import html
    from brax.training import ppo, sac
    from IPython.display import HTML, IFrame, clear_output, display
    from jax.tools import colab_tpu

    env_name = "fetch"  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
    env_fn = envs.create_fn(env_name=env_name)
    env = env_fn()
    state = env.reset(rng=jax.random.PRNGKey(seed=1))

    def visualize(sys, qps):
        """Renders a 3D visualization of the environment."""
        return HTML(html.render(sys, qps))

    # htmlrender = visualize(env.sys, [state.qp])
    html.save_html(path=f"tmp/env_render/{env_name}.html", sys=env.sys, qps=[state.qp])
