from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLHalfcheetah(CARLBraxEnv):
    env_name: str = "halfcheetah"
    asset_path: str = "envs/assets/half_cheetah.xml"
