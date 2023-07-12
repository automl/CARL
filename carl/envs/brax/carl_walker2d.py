from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLWalker2d(CARLBraxEnv):
    env_name: str = "walker2d"
    asset_path: str = "envs/assets/walker2d.xml"
