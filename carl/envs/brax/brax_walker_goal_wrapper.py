import gym
import numpy as np
from brax.io import mjcf
from etils import epath

STATE_INDICES = {
    "ant": [13, 14],
    "humanoid": [22, 23],
    "halfcheetah": [14, 15],
    "hopper": [5, 6],
    "walker2d": [8, 9],
}

DIRECTION_NAMES = {
    1: "north",
    3: "south",
    2: "east",
    4: "west",
    12: "north east",
    32: "south east",
    14: "north west",
    34: "south west",
    112: "north north east",
    332: "south south east",
    114: "north north west",
    334: "south south west",
    212: "east north east",
    232: "east south east",
    414: "west north west",
    434: "west south west",
}

directions = [
    1,  # north
    3,  # south
    2,  # east
    4,  # west
    12,
    32,
    14,
    34,
    112,
    332,
    114,
    334,
    212,
    232,
    414,
    434,
]


class BraxWalkerGoalWrapper(gym.Wrapper):
    """Adds a positional goal to brax walker envs"""

    def __init__(self, env: gym.Env, env_name: str, asset_path: str) -> None:
        super().__init__(env)
        self.env_name = env_name
        if (
            self.env_name == "humanoid"
            or self.env_name == "halfcheetah"
            or self.env_name == "hopper"
            or self.env_name == "walker2d"
        ):
            self.env._forward_reward_weight = 0
        self.context = None
        self.position = None
        self.goal_position = None
        self.goal_radius = None
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
        path = epath.resource_path("brax") / asset_path
        sys = mjcf.load(path)
        self.dt = sys.opt.timestep

    def reset(self, seed=None, options={}):
        state, info = self.env.reset(seed=seed, options=options)
        self.position = (0, 0)
        self.goal_position = (
            np.array(self.direction_values[self.context["target_direction"]])
            * self.context["target_distance"]
        )
        self.goal_radius = self.context["target_radius"]
        info["success"] = 0
        return state, info

    def step(self, action):
        state, _, te, tr, info = self.env.step(action)
        indices = STATE_INDICES[self.env_name]
        new_position = (
            np.array(list(self.position))
            + np.array([state[indices[0]], state[indices[1]]]) * self.dt
        )
        current_distance_to_goal = np.linalg.norm(self.goal_position - new_position)
        previous_distance_to_goal = np.linalg.norm(self.goal_position - self.position)
        direction_reward = max(0, previous_distance_to_goal - current_distance_to_goal)
        self.position = new_position
        if abs(current_distance_to_goal) <= self.goal_radius:
            te = True
            info["success"] = 1
        else:
            info["success"] = 0
        return state, direction_reward, te, tr, info


class BraxLanguageWrapper(gym.Wrapper):
    """Translates the context features target distance and target radius into language"""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.context = None

    def reset(self, seed=None, options={}):
        self.env.context = self.context
        state, info = self.env.reset(seed=seed, options=options)
        goal_str = self.get_goal_desc(self.context)
        if isinstance(state, dict):
            state["goal"] = goal_str
        else:
            state = {"obs": state, "goal": goal_str}
        return state, info

    def step(self, action):
        state, reward, te, tr, info = self.env.step(action)
        goal_str = self.get_goal_desc(self.context)
        if isinstance(state, dict):
            state["goal"] = goal_str
        else:
            state = {"obs": state, "goal": goal_str}
        return state, reward, te, tr, info

    def get_goal_desc(self, context):
        if "target_radius" in context.keys():
            target_distance = context["target_distance"]
            target_direction = context["target_direction"]
            target_radius = context["target_radius"]
            return f"The distance to the goal is {target_distance}m {DIRECTION_NAMES[target_direction]}. Move within {target_radius} steps of the goal."
        else:
            target_distance = context["target_distance"]
            target_direction = context["target_direction"]
            return f"Move {target_distance}m {DIRECTION_NAMES[target_direction]}."
