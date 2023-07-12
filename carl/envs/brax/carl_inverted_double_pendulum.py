from __future__ import annotations

from carl.envs.brax.carl_brax_env import CARLBraxEnv


class CARLInvertedDoublePendulum(CARLBraxEnv):
    env_name: str = "inverted_double_pendulum"
    asset_path: str = "envs/assets/inverted_double_pendulum.xml"
