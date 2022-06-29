from pathlib import Path
from typing import Dict, Any, Union
import wandb
import numpy as np
import pandas as pd

from experiments.common.utils.json_utils import lazy_json_dump, lazy_json_load


def log_contexts_wandb_traineval(train_contexts, eval_contexts):
    log_contexts_wandb(contexts=train_contexts, wandb_key="train/contexts")
    log_contexts_wandb(contexts=eval_contexts, wandb_key="eval/contexts")


def log_contexts_wandb(contexts: Dict[Any, Dict[str, Any]], wandb_key: str):
    table = wandb.Table(
        columns=sorted(contexts[list(contexts.keys())[0]].keys()),
        data=[
            [contexts[idx][key] for key in sorted(contexts[idx].keys())]
            for idx in contexts.keys()
        ],
    )
    wandb.log({wandb_key: table}, step=0)


def log_contexts_json(contexts, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    lazy_json_dump(contexts, str(path))


def load_wandb_table_to_df(path: Union[str, Path]) -> pd.DataFrame:
    data = lazy_json_load(path)
    df = pd.DataFrame(data=np.array(data["data"]), columns=data["columns"])
    return df


def load_wandb_contexts(path: Union[str, Path]) -> Dict[Any, Dict[Any, Any]]:
    data = lazy_json_load(path)
    cols = data["columns"]
    D = data["data"]
    contexts = {
        i: {
            cols[j]: v for j, v in enumerate(context_values)
        } for i, context_values in enumerate(D)
    }
    return contexts

