from typing import List

import pandas as pd


def filter_group(df: pd.DataFrame, target: List) -> pd.DataFrame:
    key = "group"
    return filterdf(key=key, df=df, target=target)


def filter_std(df: pd.DataFrame, target: List) -> pd.DataFrame:
    key = "contexts.default_sample_std_percentage"
    return filterdf(key=key, df=df, target=target)


def filter_cf(df: pd.DataFrame, target: List) -> pd.DataFrame:
    key = "contexts.context_feature_args"
    return filterdf(key=key, df=df, target=target)


def filterdf(key: str, df: pd.DataFrame, target: List) -> pd.DataFrame:
    exclude = [e for e in df[key].unique() if e not in target]
    for ex in exclude:
        ids = df[key] == ex
        df = df[~ids]
    return df