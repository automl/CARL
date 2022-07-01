from pathlib import Path
from typing import Dict, Any, Union
import wandb

from experiments.common.utils.json_utils import lazy_json_dump


def log_contexts_wandb_traineval(train_contexts, eval_contexts):
    log_contexts_wandb(contexts=train_contexts, wandb_key="train/contexts")
    log_contexts_wandb(contexts=eval_contexts, wandb_key="eval/contexts")


def log_contexts_wandb(contexts: Dict[Any, Dict[str, Any]], wandb_key: str):
    table = wandb.Table(
        columns=sorted(contexts[0].keys()),
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
