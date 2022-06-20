from typing import Union, Optional
from enum import Enum, IntEnum, auto
from pathlib import Path
import wandb

from carl.context.sampling import get_default_context_and_bounds


def log_contexts_wandb(train_contexts, eval_contexts):
    table = wandb.Table(
        columns=sorted(train_contexts[0].keys()),
        data=[
            [train_contexts[idx][key] for key in sorted(train_contexts[idx].keys())]
            for idx in train_contexts.keys()
        ],
    )
    wandb.log({"train/contexts": table}, step=0)
    eval_table = wandb.Table(
        columns=sorted(eval_contexts[0].keys()),
        data=[
            [eval_contexts[idx][key] for key in sorted(eval_contexts[idx].keys())]
            for idx in eval_contexts.keys()
        ],
    )
    wandb.log({"eval/contexts": eval_table}, step=0)


class ContextDifficulty(Enum):
    easy = auto()
    medium = auto()
    hard = auto()


class NContexts(Enum):
    small = 100
    medium = 1000
    large = 10000


class AbstractContextSampler(object):
    pass


class ContextSampler(AbstractContextSampler):
    def __init__(
            self,
            env_name: str,
            difficulty: str = "easy",
            n_samples: Union[str, int] = 100,
            out_dir: Optional[Union[str, Path]] = None,
    ):
        super(ContextSampler, self).__init__()
        self.env_name = env_name
        self.C_def, self.C_bounds = get_default_context_and_bounds(env_name=self.env_name)

        if difficulty not in ContextDifficulty.__members__:
            raise ValueError("Please specify a valid difficulty. Valid options are: ",
                  list(ContextDifficulty.__members__.keys()))
        self.difficulty = ContextDifficulty[difficulty]

        if type(n_samples) == str:
            if n_samples not in NContexts.__members__:
                raise ValueError("Please specify a valid size. Valid options are: ",
                      list(NContexts.__members__.keys()))
            self.n_samples = NContexts[n_samples]
        elif type(n_samples) == int or type(n_samples) == float:
            self.n_samples = int(n_samples)
        else:
            raise ValueError(f"`n_samples` must be of type str "
                             f"or int, got {type(n_samples)}.")

        self.out_dir = out_dir

    def log_contexts(self):
        path = Path(self.out_dir)
        path.mkdir(exist_ok=True, parents=True)



if __name__ == "__main__":
    env_name = "CARLPendulumEnv"
    cs = ContextSampler(
        env_name=env_name,
        difficulty="easy",
        n_samples=10023
    )
