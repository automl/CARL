from typing import Union, Optional

from gymnasium.envs.box2d.lunar_lander import heuristic
import gymnasium.envs.box2d.lunar_lander as lunar_lander
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from carl.envs import CARLLunarLanderEnv


def demo_heuristic_lander(
    env: Union[
        CARLLunarLanderEnv, lunar_lander.LunarLander, lunar_lander.LunarLanderContinuous
    ],
    seed: Optional[int] = None,
    render: bool = False,
) -> float:
    """
    Copied from LunarLander
    """

    total_reward = 0
    steps = 0
    if render:
        env.render()
    s = env.reset(
        seed=seed,
    )

    while True:
        a = heuristic(env, s)

        s, r, done, truncated, info = env.step(a)

        total_reward += r

        if render and steps % 20 == 0:
            still_open = env.render()

        if done or truncated:  # or steps % 20 == 0:
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    return total_reward


if __name__ == "__main__":
    env = CARLLunarLanderEnv(
        hide_context=False,
        add_gaussian_noise_to_context=True,
        gaussian_noise_std_percentage=0.1,
    )
    # env.render()  # initialize viewer. otherwise weird bug.
    # env = ll.LunarLander()
    # env = CustomLunarLanderEnv()
    for i in range(5):
        demo_heuristic_lander(env, seed=1, render=True)
    env.close()
