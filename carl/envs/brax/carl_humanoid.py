from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxHumanoid(CARLBraxEnv):
    env_name: str = "humanoid"
    asset_path: str = "envs/assets/humanoid.xml"
