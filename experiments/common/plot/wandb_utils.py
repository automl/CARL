import wandb
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf
from functools import reduce
from operator import getitem
from rich import print, inspect


def collect_metric_history(
    run: wandb.apis.public.Run, metrics: List[str]
) -> pd.DataFrame:
    """
    Collect the metric history (progress over time/steps) from a wandb run.

    Parameters
    ----------
    run : wandb.apis.public.Run
        Run object from wandb.
    metrics : List[str]
        Metrics used to track progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["_step", *metrics].


    """
    df = pd.DataFrame()
    for i, row in run.history(keys=metrics).iterrows():
        if all([metric in row for metric in metrics]):
            df = df.append(row, ignore_index=True)
    return df


def collect_metadata(run: wandb.apis.public.Run) -> Dict[str, Any]:
    df = {}
    df["_timestamp"] = run.summary["_timestamp"]
    df["name"] = run.name
    return df


def add_scalardict_to_df(scalardict: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scalar metadata to long-form DataFrame.

    Parameters
    ----------
    scalardict
    df

    Returns
    -------
    pd.DataFrame

    """
    for k, v in scalardict.items():
        df[k] = v
    return df


def collect_config(
    run: wandb.apis.public.Run, config_entries: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Filter config.

    If config_entries is None, collect all config entries.

    Parameters
    ----------
    run
    config_entries

    Returns
    -------
    Dict[str, Any]
        Config
    """
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}

    # flatten config (dot-separated keys)
    config = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]

    if config_entries is not None:
        # filter
        config = {k: v for k, v in config.items() if k in config_entries}
    return config


def get_nested_item(data, keys):
    try:
        return reduce(getitem, keys, data)
    except (KeyError, IndexError):
        return None


def load_wandb(
    project_name: str,
    df_fname: Union[str, Path],
    filters: Optional[Dict] = None,
    redownload: bool = False,
    metrics: Optional[List[str]] = None,
    config_entries: Optional[List[str]] = None,
) -> pd.DataFrame:
    df_fname = Path(df_fname)
    df_fname.parent.mkdir(parents=True, exist_ok=True)

    if not df_fname.is_file() or redownload:
        api = wandb.Api()
        runs = api.runs(project_name, filters=filters)

        dfs = []
        runs = list(runs)
        for run in tqdm(runs):
            # print(vars(run))
            # inspect(run)
            df = collect_metric_history(run=run, metrics=metrics)
            metadata = collect_metadata(run=run)
            df = add_scalardict_to_df(scalardict=metadata, df=df)
            config = collect_config(run=run, config_entries=config_entries)
            df = add_scalardict_to_df(scalardict=config, df=df)
            if len(df) > 0:
                dfs.append(df)
        data = pd.concat(dfs)
        data.reset_index(inplace=True, drop=True)
        data.to_csv(df_fname, index=False)
    else:
        data = pd.read_csv(df_fname)
        if "Unnamed: 0" in data:
            del data["Unnamed: 0"]

    return data
