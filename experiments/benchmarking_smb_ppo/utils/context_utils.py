from typing import Any, Dict, List, Union

from omegaconf import DictConfig
from .context_sampling import ContextSampler


def parse_values(val: Union[str, List[Any]]):
    if isinstance(val, str) and val.startswith("range"):
        _, start, stop = val.split("_")
        return list(range(int(start), int(stop)))
    elif isinstance(val, list):
        return val
    else:
        return []


def parse_contexts(cfg: DictConfig):
    contexts = []
    for context_feature, context_value_def in cfg.per_context_features.items():
        context_values = parse_values(context_value_def)
        for context_value in context_values:
            contexts.append({**cfg.constant_features, **{context_feature: context_value}})
    return contexts


def get_contexts(cfg, is_eval=False):
    if is_eval:
        if cfg.eval_contexts:
            return parse_contexts(cfg.eval_contexts)
    else:
        if cfg.contexts:
            return parse_contexts(cfg.contexts)
    return ContextSampler(**cfg.context_sampler).sample_contexts()
