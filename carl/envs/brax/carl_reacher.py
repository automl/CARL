from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLReacher(CARLBraxEnv):
    env_name: str = "reacher"
    asset_path: str = "envs/assets/reacher.xml"
