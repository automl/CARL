# Preprocessing from stable baselines zoo
# Source: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/exp_manager.py
from experiments.common.utils.rl_baselines3_zoo.utils.utils import get_wrapper_class
from experiments.common.utils.rl_baselines3_zoo.utils.exp_manager import ExperimentManager
from typing import Any, Dict


def preprocess_hyperparams(hyperparams: Dict[str, Any]):
    # Convert schedule strings to objects
    hyperparams = ExperimentManager._preprocess_schedules(hyperparams)

    # Pre-process train_freq
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    # Pre-process policy/buffer keyword arguments
    # Convert to python object if needed
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    # Delete keys so the dict can be pass to the model constructor
    if "n_envs" in hyperparams.keys():
        del hyperparams["n_envs"]
    del hyperparams["n_timesteps"]

    if "frame_stack" in hyperparams.keys():
        del hyperparams["frame_stack"]

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    if "callback" in hyperparams.keys():
        del hyperparams["callback"]
    if "noise_type" in hyperparams.keys():
        del hyperparams["noise_type"]
        del hyperparams["noise_std"]

    normalize_kwargs = None
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        normalize_kwargs = {}
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True

        if "gamma" in hyperparams:
            normalize_kwargs["gamma"] = hyperparams["gamma"]

        del hyperparams["normalize"]
        normalize_kwargs["normalize"] = normalize

    return hyperparams, env_wrapper, normalize_kwargs
