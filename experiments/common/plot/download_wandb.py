import wandb
from tqdm import tqdm
import pandas as pd
from pathlib import Path

df_fname = Path("data/alldata.csv")
df_fname.parent.mkdir(parents=True, exist_ok=True)


api = wandb.Api()
runs = api.runs(
    "tnt/carl",
    filters={
        "state": "finished",
    },
)
dfs = []
config_entries = [
    "group",
    "contexts.context_feature_args",
    "carl.state_context_features",
    "contexts.default_sample_std_percentage",
    "seed"
]
metrics = ["eval/return"]
groups = {
    "hidden_context": "hidden",
    "concat_context": "concat",
    "context_gating": "gating",
    "context_encoder": "encoder"
}
runs = list(runs)
for run in tqdm(runs):
    df = pd.DataFrame()
    for i, row in run.history(keys=metrics).iterrows():
        if all([metric in row for metric in metrics]):
            df = df.append(row, ignore_index=True)
    df["timestamp"] = run.summary["_timestamp"]
    df["name"] = run.name
    for entry in config_entries:
        entry_list = entry.split(".")
        config_entry = run.config
        for e in entry_list:
            config_entry = config_entry[e]
        if isinstance(config_entry, (list, tuple)):
            config_entry = ", ".join(config_entry)
        if entry == "group":
            df[entry] = groups[config_entry]
        else:
            df[entry] = config_entry
    if len(df) > 1:
        dfs.append(df)
df = pd.concat(dfs)
df.reset_index(inplace=True)
df.to_csv(df_fname, index=False)