from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLBraxInvertedPendulum(CARLBraxEnv):
    env_name: str = "inverted_pendulum"
    asset_path: str = "envs/assets/inverted_pendulum.xml"
