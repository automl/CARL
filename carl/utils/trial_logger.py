from typing import Any, Dict, Union

import argparse
from pathlib import Path

import configargparse
import pandas as pd


class TrialLogger(object):
    """
    Holds all train arguments and sets up logging directory and stables baselines
    logging, writes trial setup and writes context feature history.

    Following logging happens at the corresponding events:
    after each step:
        reward (progress.csv) (StableBaselines logger)
        step (progress.csv) (StableBaselines logger)

    after each episode:
        context (context_history.csv) (TrialLogger)
        episode (context_history.csv) (TrialLogger)
        step (context_history.csv) (TrialLogger)

    once on train start:
        experiment config (env, agent, seed, set of contexts) (TrialLogger)
        hyperparameters

    """

    def __init__(
        self,
        logdir: Union[str, Path],
        parser: configargparse.ArgParser,
        trial_setup_args: argparse.Namespace,
        add_context_feature_names_to_logdir: bool = False,
    ):
        """

        Parameters
        ----------
        logdir: Union[str, Path]
            Base logging directory. The actual logging directory, accessible via self.logdir,
            is logdir / "{agent}_{seed}".
            Agent and seed are provided via trial_setup_args.
            If add_context_feature_names_to_logdir is True,
            the logging directory will be logdir / context_feature_dirname /f"{agent}_{seed}".
            context_feature_dirname are all context feature names provided via
            trial_setup_args.context_feature_args joined by "__".
        parser: configargparse.ArgParser
            Argument parser containing all arguments from runscript. Needed to write
            trial setup file.
        trial_setup_args: argparse.Namespace
            Parsed arguments from parser. Arguments are supposed to be parsed before in case
            new arguments are added via some external logic.
        add_context_feature_names_to_logdir: bool, False
            See logdir for effect.

        """
        self.parser = parser
        seed = trial_setup_args.seed
        agent = trial_setup_args.agent
        if add_context_feature_names_to_logdir:
            context_feature_args = trial_setup_args.context_feature_args
            names = [
                n for n in context_feature_args if "std" not in n and "mean" not in n
            ]  # TODO make sure to exclude numbers
            context_feature_dirname = "default"
            if names:
                context_feature_dirname = (
                    names[0] if len(names) == 1 else "__".join(names)
                )
            self.logdir = Path(logdir) / context_feature_dirname / f"{agent}_{seed}"
        else:
            self.logdir = Path(logdir) / f"{agent}_{seed}"
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.trial_setup_args = trial_setup_args
        self.trial_setup_fn = self.logdir / "trial_setup.ini"

        self.context_history_fn = self.logdir / "context_history.csv"
        self.prepared_context_history_file = False

    def write_trial_setup(self) -> None:
        """
        Write trial setup to file with path logdir / "trial_setup.ini".

        Returns
        -------
        None

        """
        output_file_paths = [str(self.trial_setup_fn)]
        self.parser.write_config_file(
            parsed_namespace=self.trial_setup_args, output_file_paths=output_file_paths
        )

    def write_context(self, episode: int, step: int, context: Dict[Any, Any]):
        """
        Context will be written to csv file (logdir / "context_history.csv").

        The format is as follows:
            episode,step,context_feature_0,context_feature_1,...,context_feature_n
            0,1,345345,234234,...,234234

        Parameters
        ----------
        episode: int
            Episode.
        step: int
            Timestep.
        context: Dict[Any, Any]
            Keys: Context features names/ids, values: context feature values.

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
            mode=mode,
        )
