from functools import partial

from gym.envs.registration import register

from .mario_env import MarioEnv

for level in range(1, 16):
    level_file = f"lvl-{level}.txt"
    env_id = f"MarioLevel{level}-v0"
    register(id=env_id, entry_point=partial(MarioEnv, levels=[level_file]))

register(
    id="MarioLevelBase-v0",
    entry_point=partial(MarioEnv, levels=["lvl-1.txt", "lvl-2.txt", "lvl-3.txt"]),
)


register(
    id="MarioLevelAll-v0",
    entry_point=partial(MarioEnv, levels=[f"lvl-{i}.txt" for i in range(1, 16)]),
)
