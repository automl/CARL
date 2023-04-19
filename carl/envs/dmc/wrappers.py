from typing import Any, Optional, Tuple, TypeVar, Union

import dm_env  # type: ignore
import gymnasium as gym
import numpy as np
from dm_env import StepType
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def get_shape(shape: tuple) -> tuple:
    """
    Get shape of array or scalar.

    If scalar (shape = ()), return (1,).

    Parameters
    ----------
    shape: tuple
        Shape of array, can be empty tuple

    Returns
    -------
    Shape: Same as before if not empty, else (1,)
    """
    return shape if shape else (1,)


class MujocoToGymWrapper(gym.Env):
    def __init__(self, env: dm_env) -> None:
        # TODO set seeds
        self.env = env

        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=action_spec.dtype
        )

        obs_spec = self.env.observation_spec()
        # obs_spaces = {
        #     k: spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
        #     for k, v in obs_spec.items()
        # }
        # self.observation_space = spaces.Dict(spaces=obs_spaces)
        # TODO add support for Dict Spaces in CARLEnv (later)
        shapes = [int(np.sum([get_shape(v.shape) for v in obs_spec.values()]))]
        lows = np.array([-np.inf] * shapes[0])
        highs = np.array([np.inf] * shapes[0])
        dtype = np.unique([[v.dtype for v in obs_spec.values()]])[0]
        self.observation_space = spaces.Box(
            low=lows, high=highs, shape=shapes, dtype=dtype
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            terminated (bool): whether termination condition is reached
            truncated (bool): whether the episode has ended due to time limit
            info (dict): contains auxiliary diagnostic information
                            (helpful for debugging, logging, and sometimes learning)
        """
        timestep = self.env.step(action=action)
        step_type: StepType = timestep.step_type
        reward = timestep.reward
        discount = timestep.discount
        observation = timestep.observation["observations"]
        info = {"step_type": step_type, "discount": discount}
        done = step_type == StepType.LAST
        return observation, reward, False, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super(MujocoToGymWrapper, self).reset(seed=seed, options=options)
        timestep = self.env.reset()
        if isinstance(self.observation_space, spaces.Box):
            observation = timestep.observation["observations"]
        else:
            raise NotImplementedError
        return observation, {}

    def render(
        self, mode: str = "human", camera_id: int = 0, **kwargs: Any
    ) -> np.ndarray:
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render_modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            camera_id
            kwargs: Keyword arguments for dm_control.mujoco.engine.Physics.render

        Example:

        class MyEnv(Env):
            metadata = {'render_modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        # TODO render mujoco human version

        if mode == "human":
            raise NotImplementedError
        elif mode == "rgb_array":
            return self.env.physics.render(camera_id=camera_id, **kwargs)
        else:
            raise NotImplementedError
