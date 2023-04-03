# flake8: noqa: F401
# type: ignore
import matplotlib.pyplot as plt

from carl.envs import CARLDmcFishEnv
from carl.envs import CARLDmcFishEnv_defaults as fish_default
from carl.envs import CARLDmcFishEnv_mask as fish_mask
from carl.envs import CARLDmcQuadrupedEnv
from carl.envs import CARLDmcQuadrupedEnv_defaults as quadruped_default
from carl.envs import CARLDmcQuadrupedEnv_mask as quadruped_mask
from carl.envs import CARLDmcWalkerEnv
from carl.envs import CARLDmcWalkerEnv_defaults as walker_default
from carl.envs import CARLDmcWalkerEnv_mask as walker_mask
from carl.envs import CARLDmcFingerEnv

import os 
os.environ["MUJOCO_GL"] = "glfw"
# os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == "__main__":
    # Load one task:
    stronger_act = walker_default.copy()
    stronger_act["actuator_strength"] = walker_default["actuator_strength"] * 2
    contexts = {0: stronger_act}

    # stronger_act = quadruped_default.copy()
    # stronger_act["actuator_strength"] = quadruped_default["actuator_strength"]*2
    # contexts = {0: stronger_act}
    # carl_env = CARLDmcQuadrupedEnv(task="fetch_context", contexts=contexts, context_mask=quadruped_mask, hide_context=False)

    # contexts = {0: fish_default}
    #     carl_env = CARLDmcFishEnv(task="upright_context", contexts=contexts, context_mask=fish_mask, hide_context=False)
    carl_env = CARLDmcWalkerEnv(
        task="run_context",
        contexts=contexts,
        context_mask=walker_mask,
        hide_context=False,
        dict_observation_space=True,
    )
    carl_env = CARLDmcFingerEnv()
    # carl_env = CARLDmcWalkerEnv()
    # carl_env = CARLDmcQuadrupedEnv()
    # carl_env = CARLDmcFishEnv()
    action = carl_env.action_space.sample()
    s = carl_env.reset()
    state, reward, done, info = carl_env.step(action=action)
    print("state", state, type(state))

    def render(env, **render_kwargs):
        frame = carl_env.render(mode="rgb_array", **render_kwargs)
        plt.imshow(frame)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"dm_render_{type(env).__name__}.png", dpi=300, bbox_inches='tight',transparent=True, pad_inches=0)

    s = carl_env.reset()
    render(
        carl_env, 
        camera_id=1,
        height=400,
        width=400,
    )
    
    action = carl_env.action_space.sample()
    state, reward, done, info = carl_env.step(action=action)
    print("state", state, type(state))

    # s = carl_env.reset()
    # done = False
    # i = 0
    # while not done:
    #     action = carl_env.action_space.sample()
    #     state, reward, done, info = carl_env.step(action=action)
    #     print(state, action, reward, done)
    #     i += 1
    # print(i)
