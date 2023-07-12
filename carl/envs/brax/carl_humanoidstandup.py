from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxHumanoidStandup(CARLBraxEnv):
    env_name: str = "humanoidstandup"
    asset_path: str = "envs/assets/humanoidstandup.xml"
