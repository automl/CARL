from src.experiments.evaluation_protocol.evaluation_protocol import ContextFeature


def get_context_features(env_name):
    if env_name == "CARLCartPoleEnv":
        cf0 = ContextFeature("gravity", 9., 9.5, 10., 11.)
        cf1 = ContextFeature("pole_length", 0.4, 0.5, 0.6, 0.8)
    else:
        raise NotImplementedError
    context_features = [cf0, cf1]
    return context_features


def get_solved_threshold(env_name):
    thresh = None
    if env_name == "CARLCartPoleEnv":
        thresh = 195
    elif env_name == "CARLPendulumEnv":
        thresh = -175
    return thresh