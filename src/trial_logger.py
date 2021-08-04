"""
after each step:
    reward (progress.csv)
    step (progress.csv)

after each episode:
    context (context_history.csv)
    episode (context_history.csv)
    step (context_history.csv)

once on train start:
    experiment config (env, agent, seed, set of contexts)
    hyperparameters


stablebaselines tensorboard logger logs:
    - rollout/ep_len_mean
    - rollout/ep_rew_mean
    - time/fps
    - train/approx_kl
    - train/clip_fraction
    - train/clip_range
    - train/entropy_loss
    - train/explained_variance
    - train/learning_rate
    - train/loss
    - train/policy_gradient_loss
    - train/value_loss
"""
from typing import Union, Optional
from pathlib import Path
import argparse
import configargparse
import pandas as pd

from stable_baselines3.common.logger import configure


class TrialLogger(object):
    def __init__(
            self,
            logdir: Union[str, Path],
            parser: configargparse.ArgParser,
            trial_setup_args: Optional[argparse.Namespace] = None,
            add_context_feature_names_to_logdir: bool = False,
    ):
        self.parser = parser
        seed = trial_setup_args.seed
        agent = trial_setup_args.agent
        if add_context_feature_names_to_logdir:
            context_feature_args = trial_setup_args.context_feature_args
            names = [n for n in context_feature_args if "std" not in n and "mean" not in n]
            context_feature_dirname = "default"
            if names:
                context_feature_dirname = names[0] if len(names) == 1 else "__".join(names)
            self.logdir = Path(logdir) / context_feature_dirname/f"{agent}_{seed}"
        else:
            self.logdir = Path(logdir) / f"{agent}_{seed}"
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.trial_setup_args = trial_setup_args
        self.trial_setup_fn = self.logdir / "trial_setup.ini"

        self.context_history_fn = self.logdir / "context_history.csv"
        self.prepared_context_history_file = False

        self.stable_baselines_logger = configure(str(self.logdir), ["stdout", "csv", "tensorboard"])

    def write_trial_setup(self):
        output_file_paths = [str(self.trial_setup_fn)]
        self.parser.write_config_file(parsed_namespace=self.trial_setup_args, output_file_paths=output_file_paths)

    def write_context(self, episode: int, step: int, context: dict):
        """
        Context will be written to csv file.

        The format is as follows:
            episode,step,context_feature_0,context_feature_1,...,context_feature_n
            0,1,345345,234234,...,234234

        Parameters
        ----------
        episode: int
        step: int
        context: dict

        Returns
        -------
        None
        """
        columns = ["episode", "step"] + list(context.keys())
        values = [episode, step] + list(context.values())
        df = pd.DataFrame(values).T
        df.columns = columns

        write_header = False
        mode = "a"
        if not self.prepared_context_history_file:
            write_header = True
            mode = "w"
            self.prepared_context_history_file = True

        df.to_csv(
            path_or_buf=self.context_history_fn,
            sep=",",
            header=write_header,
            index=False,
            mode=mode
        )



