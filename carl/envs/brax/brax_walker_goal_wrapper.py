import gym
import numpy as np

STATE_INDICES = {
    "CARLAnt": [13, 14],
    "CARLHumanoid": [22, 23],
    "CARLHalfcheetah": [14, 15],
}


class BraxWalkerGoalWrapper(gym.Wrapper):
    """Adds a positional goal to brax walker envs"""

    def __init__(self, env) -> None:
        super().__init__(env)
        if (
            self.env.__class__.__name__ == "CARLHumanoid"
            or self.env.__class__.__name__ == "CARLHalfcheetah"
        ):
            self.env._forward_reward_weight = 0
        self.position = None
        self.goal_position = None
        self.direction_values = {
            3: [0, -1],
            1: [0, 1],
            2: [1, 0],
            4: [-1, 0],
            34: [-np.sqrt(0.5), -np.sqrt(0.5)],
            14: [-np.sqrt(0.5), np.sqrt(0.5)],
            32: [np.sqrt(0.5), -np.sqrt(0.5)],
            12: [np.sqrt(0.5), np.sqrt(0.5)],
            334: [
                -np.cos(22.5 * np.pi / 180),
                -np.sin(22.5 * np.pi / 180),
            ],
            434: [
                -np.sin(22.5 * np.pi / 180),
                -np.cos(22.5 * np.pi / 180),
            ],
            114: [
                -np.cos(22.5 * np.pi / 180),
                np.sin(22.5 * np.pi / 180),
            ],
            414: [
                -np.sin(22.5 * np.pi / 180),
                np.cos(22.5 * np.pi / 180),
            ],
            332: [
                np.cos(22.5 * np.pi / 180),
                -np.sin(22.5 * np.pi / 180),
            ],
            232: [
                np.sin(22.5 * np.pi / 180),
                -np.cos(22.5 * np.pi / 180),
            ],
            112: [
                np.cos(22.5 * np.pi / 180),
                np.sin(22.5 * np.pi / 180),
            ],
            212: [np.sin(22.5 * np.pi / 180), np.cos(22.5 * np.pi / 180)],
        }

    def reset(self, return_info=False):
        state, info = self.env.reset(info=True)
        self.position = (0, 0)
        self.goal_position = (
            np.array(self.direction_values[self.context["target_direction"]])
            * self.context["target_distance"]
        )
        if return_info:
            return state, info
        else:
            return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        indices = STATE_INDICES[self.env.__class__.__name__]
        new_position = np.array(list(self.position)) + np.array(
            [state[indices[0]], state[indices[1]]]
        )
        current_distance_to_goal = np.linalg.norm(self.goal_position - new_position)
        previous_distance_to_goal = np.linalg.norm(self.goal_position - self.position)
        direction_reward = max(0, previous_distance_to_goal - current_distance_to_goal)
        if self.env.__class__.__name__ == "CARLAnt":
            # Since we can't set the forward reward to 0 here, we simply increase the reward range
            direction_reward = direction_reward * 10
        augmented_reward = reward + direction_reward
        self.position = new_position
        return state, augmented_reward, done, info
