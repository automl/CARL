from __future__ import annotations

import ast
from typing import Dict
from omegaconf import DictConfig, OmegaConf


def recover_traincfg_from_wandb(
    fn_wbcfg: str, to_dict: bool = False
) -> DictConfig | Dict | None:
    wbcfg = OmegaConf.load(fn_wbcfg)
    if not "traincfg" in wbcfg:
        return None
    traincfg = wbcfg.traincfg
    traincfg = OmegaConf.to_container(cfg=traincfg, resolve=False, enum_to_str=True)[
        "value"
    ]
    traincfg = ast.literal_eval(traincfg)
    traincfg = OmegaConf.create(traincfg)
    if to_dict:
        traincfg = OmegaConf.to_container(cfg=traincfg, resolve=True, enum_to_str=True)
    return traincfg
