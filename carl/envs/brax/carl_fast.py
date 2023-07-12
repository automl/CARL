from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxFast(CARLBraxEnv):
    env_name: str = "fast"
    asset_path: str = "envs/assets/fast.xml"
