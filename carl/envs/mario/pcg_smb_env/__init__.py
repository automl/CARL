from functools import partial

from gym.envs.registration import register, registry

from .mario_env import MarioEnv
from .toadgan.toad_gan import generate_initial_noise, generate_level

for level in range(1, 16):
    level_file = f"lvl-{level}.txt"
    env_id = f"MarioLevel{level}-v0"
    if env_id not in registry:
        register(id=env_id, entry_point=partial(MarioEnv, levels=[level_file]))

if "MarioLevelBase-v0" not in registry:
    register(
        id="MarioLevelBase-v0",
        entry_point=partial(MarioEnv, levels=["lvl-1.txt", "lvl-2.txt", "lvl-3.txt"]),
    )

if "MarioLevelAll-v0" not in registry:
    register(
        id="MarioLevelAll-v0",
        entry_point=partial(MarioEnv, levels=[f"lvl-{i}.txt" for i in range(1, 16)]),
    )
