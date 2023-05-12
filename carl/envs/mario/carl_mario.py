from collections import defaultdict
import os
from typing import Dict, List, Optional, Union

from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.mario.carl_mario_definitions import (
    DEFAULT_CONTEXT,
    INITIAL_HEIGHT,
    INITIAL_WIDTH,
    CONTEXT_BOUNDS  # noqa: F401
)
from carl.envs.mario.pcg_smb_env.utils import load_level
from carl.utils.trial_logger import TrialLogger
from carl.utils.types import Context, Contexts
from carl.envs.mario.pcg_smb_env import MarioEnv, generate_level


class CARLMarioEnv(CARLEnv):
    def __init__(
        self,
        env=MarioEnv(levels=[]),
        contexts: Contexts = {},
        hide_context: bool = True,
        add_gaussian_noise_to_context: bool = False,
        gaussian_noise_std_percentage: float = 0.05,
        logger: Optional[TrialLogger] = None,
        scale_context_features: str = "no",
        default_context: Optional[Context] = DEFAULT_CONTEXT,
        state_context_features: Optional[List[str]] = None,
        context_mask: Optional[List[str]] = None,
        dict_observation_space: bool = False,
        context_selector: Optional[
            Union[AbstractSelector, type[AbstractSelector]]
        ] = None,
        context_selector_kwargs: Optional[Dict] = None,
    ):
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            hide_context=True,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger,
            scale_context_features="no",
            default_context=default_context,
            dict_observation_space=dict_observation_space,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            context_mask=context_mask,
        )
        self.levels_per_context: Dict[str, List[str]] = defaultdict(list)
        os.makedirs("levels", exist_ok=True)
        self._update_context()

    def _check_if_level_exists(self, context_hash: str):
        """
        Check if the level has already been generated in the levels directory.
        """
        return f"{context_hash}.txt" in os.listdir("levels")

    def _update_context(self) -> None:
        if len(self.levels_per_context) == 0:
            for context_key, context in self.contexts.items():
                context_hash = f"{context_key}_" + "-".join([f"{k}={v}" for k, v in context.items()])
                level_path = os.path.join("levels", f"{context_hash}.txt")
                self.levels_per_context[context_key].append(
                    level_path
                )
                if self._check_if_level_exists(context_hash):
                    continue
                level, initial_noise = generate_level(
                    width=INITIAL_WIDTH,
                    height=INITIAL_HEIGHT,
                    level_index=context["level_index"],
                    seed=context["noise"],
                    filter_unplayable=True,
                )
                self.contexts[context_key]["noise"] = initial_noise
                with open(level_path, "w") as f:
                    f.write(level)
        self.env.mario_state = self.context["mario_state"]
        self.env.mario_inertia = self.context["mario_inertia"]
        self.env.levels = [load_level(level) for level in self.levels_per_context[self.context_key]]
