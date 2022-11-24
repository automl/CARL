import gym


class BraxLanguageWrapper(gym.Wrapper):
    """Translates the context features target distance and target radius into language"""

    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, return_info=False):
        state, info = self.env.reset(info=True)
        goal_str = self.get_goal_desc(info["context"])
        extended_state = {"env_state": state, "goal": goal_str}
        if return_info:
            return extended_state, info
        else:
            return extended_state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        goal_str = self.get_goal_desc(info["context"])
        extended_state = {"env_state": state, "goal": goal_str}
        return extended_state, reward, done, info

    def get_goal_desc(self, context):
        if "target_radius" in context.keys():
            target_distance = context["target_distance"]
            target_radius = context["target_radius"]
            return f"The distance to the goal is {target_distance} steps. Move within {target_radius} steps of the goal."
        else:
            target_distance = context["target_distance"]
            target_direction = context["target_direction"]
            return f"Move {target_distance} steps {target_direction}."
