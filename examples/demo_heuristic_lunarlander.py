from typing import Union, Optional

from gymnasium.envs.box2d.lunar_lander import heuristic
import gymnasium.envs.box2d.lunar_lander as lunar_lander
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from carl.envs import CARLLunarLander


def demo_heuristic_lander(
    env: Union[
        CARLLunarLander, lunar_lander.LunarLander, lunar_lander.LunarLanderContinuous
    ],
    seed: Optional[int] = None,
    render: bool = False,
) -> float:
    """
    Copied from LunarLander
    """

    total_reward = 0
    steps = 0
    s, info = env.reset(
        seed=seed,
    )
    if render:
        env.render()

    while True:
        a = heuristic(env, s["obs"])

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
    CARLLunarLander.render_mode = "human"
    env = CARLLunarLander(
        # You can play with different gravity values here
        contexts={0: {"GRAVITY_Y": -10}}
    )
    for i in range(3):
        demo_heuristic_lander(env, seed=i, render=True)
    env.close()
