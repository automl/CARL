from typing import Dict, List, Optional

import gym
import numpy as np
from src.envs.mario.mario_env import MarioEnv
from src.envs.mario.toad_gan import generate_initial_noise, generate_level
from src.envs.carl_env import CARLEnv
from src.training.trial_logger import TrialLogger

INITIAL_WIDTH = 100
INITIAL_LEVEL_INDEX = 0
INITIAL_HEIGHT = 16
DEFAULT_CONTEXT = {
    "level_index": INITIAL_LEVEL_INDEX,
    "noise": generate_initial_noise(INITIAL_WIDTH, INITIAL_HEIGHT, INITIAL_LEVEL_INDEX),
    "mario_state": 0
}

CONTEXT_BOUNDS = {
    "level_index": (None, None, "categorical", np.arange(0, 14)),
    "noise": (-1.0, 1.0, float),
    "mario_state": (None, None, "categorical", [0, 1, 2]),
}
CATEGORICAL_CONTEXT_FEATURES = ["level_index", "mario_state"]


class CARLMarioEnv(CARLEnv):
    def __init__(
        self,
        env: gym.Env = MarioEnv(levels=[]),
        contexts: Dict[int, Dict] = {},
        instance_mode: str = "rr",
        hide_context: bool = False,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.05,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Dict] = DEFAULT_CONTEXT,
        state_context_features: Optional[List[str]] = None,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=True,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features="no",
            default_context=default_context,
        )

    def _update_context(self):
        level = generate_level(
            width=INITIAL_WIDTH,
            height=INITIAL_HEIGHT,
            level_index=self.context["level_index"],
            initial_noise=self.context["noise"],
            filter_unplayable=False
        )
        self.env.mario_state = self.context["mario_state"]
        self.env.levels = [level]


if __name__ == "__main__":
    env = CARLMarioEnv(env=MarioEnv(levels=[], visual=True))
    max_episodes = 3
    record_video = True
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/home/schubert/Dropbox/video-test", force=True, video_callable=lambda _: True)
    episode = 0
    while episode < max_episodes:
        total_reward = 0.0
        steps = 0
        env.reset()
        level_img = env.unwrapped.render_current_level()
        level_img.save(f"/home/schubert/Dropbox/video-test/level_{episode}.png")
        while True:
            a = env.action_space.sample()
            s, r, done, info = env.step(a)
            total_reward += r
            env.render()
            steps += 1
            if done:
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                episode += 1
                break
    env.close()
