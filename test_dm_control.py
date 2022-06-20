import imp
from carl.envs.dmc.carl_dm_cartpole import CARLDmcCartPoleEnv
from carl.envs.dmc.carl_dm_walker import CARLDmcWalkerEnv
from carl.envs.dmc.carl_dm_quadruped import CARLDmcQuadrupedEnv
from carl.envs.dmc.carl_dm_fish import CARLDmcFishEnv
from carl.envs.classic_control import CARLCartPoleEnv
from carl.envs import CARLDmcCartPoleEnv_defaults as cartpole_default
from carl.envs import CARLDmcWalkerEnv_defaults as walker_default
from carl.envs import CARLDmcQuadrupedEnv_defaults as quadruped_default
from carl.envs import CARLDmcFishEnv_defaults as fish_default
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load one task:

    # longer_pole = cartpole_default.copy()
    # longer_pole["pole_length"] = cartpole_default["pole_length"]*2
    # contexts = {0: longer_pole}
    # carl_env = CARLDmcCartPoleEnv(task="swingup_context", contexts=contexts, hide_context=False)
    
    stronger_act = walker_default.copy()
    stronger_act["actuator_strength"] = walker_default["actuator_strength"]*2
    contexts = {0: stronger_act}
    carl_env = CARLDmcWalkerEnv(task="stand_context", contexts=contexts, hide_context=False)

    # stronger_act = walker_default.copy()
    # stronger_act["actuator_strength"] = quadruped_default["actuator_strength"]*2
    # contexts = {0: stronger_act}
    # carl_env = CARLDmcQuadrupedEnv(task="walk_context", contexts=contexts, hide_context=False)

    # contexts = {0: fish_default}
    # carl_env = CARLDmcFishEnv(task="swim_context", contexts=contexts, hide_context=False)
    render = lambda : plt.imshow(carl_env.render(mode='rgb_array'))
    s = carl_env.reset()
    render()
    plt.savefig("asdf_dm.png")
    action = carl_env.action_space.sample()
    state, reward, done, info = carl_env.step(action=action)
    print("state", state, type(state))
    assert False

    s = carl_env.reset()
    done = False
    i = 0
    while not done:
        action = carl_env.action_space.sample()
        state, reward, done, info = carl_env.step(action=action)
        print(state, action, reward, done)
        i += 1
    print(i)
