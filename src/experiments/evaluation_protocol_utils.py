from pathlib import Path
from typing import Optional, Dict, List, Union

import numpy as np
import pandas as pd

from src.context.sampling import sample_contexts
from src.experiments.evaluation_protocol import ContextFeature, EvaluationProtocol
from src.utils.json_utils import lazy_json_dump


def get_context_features(env_name):
    if env_name == "CARLCartPoleEnv":
        cf0 = ContextFeature("gravity", 9., 9.5, 10., 11.)
        cf1 = ContextFeature("pole_length", 0.4, 0.5, 0.6, 0.8)
    else:
        raise NotImplementedError
    context_features = [cf0, cf1]
    return context_features


def get_ep_contexts(env_name, n_contexts, seed, mode) -> Dict[str, pd.DataFrame]:
    context_features = get_context_features(env_name=env_name)
    ep = EvaluationProtocol(context_features=context_features, mode=mode, seed=seed)
    contexts_train = ep.create_train_contexts(n=n_contexts)
    contexts_ES = ep.create_contexts_extrapolation_single(n=n_contexts)  # covers two quadrants
    contexts_EA = ep.create_contexts_extrapolation_all(n=n_contexts)
    contexts_I = ep.create_contexts_interpolation(n=n_contexts, contexts_forbidden=contexts_train)
    contexts_IC = ep.create_contexts_interpolation_combinatorial(n=n_contexts, contexts_forbidden=contexts_train)
    contexts_dict = {
        "train": contexts_train,
        "test_interpolation": contexts_I,
        "test_interpolation_combinatorial": contexts_IC,
        "test_extrapolation_single": contexts_ES,
        "test_extrapolation_all": contexts_EA,
    }
    return contexts_dict


def merge_contexts(ep_contexts: pd.DataFrame, contexts: Dict) -> Optional[Dict]:
    if len(ep_contexts) == 0:
        return None
    for i in range(len(contexts)):
        C_ep = ep_contexts.iloc[i].to_dict()
        contexts[i].update(C_ep)
    return contexts


def get_train_contexts(env_name, seed, n_contexts, mode) -> Optional[Dict]:
    contexts_dict = get_ep_contexts(env_name=env_name, seed=seed, n_contexts=n_contexts, mode=mode)
    contexts = sample_contexts(env_name, [], n_contexts)
    contexts_train_ep = contexts_dict["train"]
    contexts_train = merge_contexts(ep_contexts=contexts_train_ep, contexts=contexts)
    return contexts_train


def create_ep_contexts_LUT(
        env_name: str,
        n_contexts: int,
        seeds: List[int],
        modes: List[str],
        contexts_LUT_fn: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    index_col = ["mode", "seed", "context_distribution_type", "instance_id"]
    contexts_LUT = []
    for mode in modes:
        for seed in seeds:
            context_dict = get_ep_contexts(env_name=env_name, n_contexts=n_contexts, seed=seed, mode=mode)
            for context_distribution_type, contexts in context_dict.items():
                if type(contexts) == list and len(contexts) == 0:
                    continue
                arrays = [
                    [mode] * len(contexts),
                    [seed] * len(contexts),
                    [context_distribution_type] * len(contexts),
                    np.arange(0, len(contexts))
                ]
                tuples = list(zip(*arrays))
                index = pd.MultiIndex.from_tuples(tuples, names=index_col)
                contexts.index = index
                contexts_LUT.append(contexts)

    contexts_LUT = pd.concat(contexts_LUT)
    if contexts_LUT_fn is not None:
        contexts_LUT.to_csv(str(contexts_LUT_fn))

    return contexts_LUT


def read_ep_contexts_LUT(csv_filename: Union[str, Path]) -> pd.DataFrame:
    index_col = ["mode", "seed", "context_distribution_type", "instance_id"]
    contexts_LUT = pd.read_csv(str(csv_filename), index_col=index_col)
    return contexts_LUT


if __name__ == '__main__':
    n_contexts = 100
    seed = 1
    mode = "A"
    env_name = "CARLCartPoleEnv"

    seeds = [0, 1, 2, 3, 4]
    modes = ["A", "B", "C"]

    fnames = []
    for seed in seeds:
        for mode in modes:
            contexts_train = get_train_contexts(env_name=env_name, seed=seed, mode=mode, n_contexts=n_contexts)

            path = Path(__file__).parent / "data" / "evaluation_protocol"
            path.mkdir(parents=True, exist_ok=True)
            fname = path / f"contexts_train-env_{env_name}-mode_{mode}-seed_{seed}.json"
            lazy_json_dump(contexts_train, fname)
            fnames.append(fname)

        break