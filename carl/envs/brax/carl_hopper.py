from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLHopper(CARLBraxEnv):
    env_name: str = "hopper"
    asset_path: str = "envs/assets/hopper.xml"
