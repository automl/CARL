from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxAnt(CARLBraxEnv):
    env_name: str = "ant"
    asset_path: str = "envs/assets/ant.xml"
