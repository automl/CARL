import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm
from pathlib import Path


def load_data_context_gating(fname: Path, download: bool = False, env: str = "CARLPendulumEnv"):
    if not fname.is_file() or download:
        api = wandb.Api()
        runs = api.runs(
            "tnt/carl",
            filters={
                "config.contexts.default_sample_std_percentage": {"$in": [0.1, 0.25, 0.5]},
                "state":  {"$nin": ["crashed"]},
                "config.env": env,
                # "group": {"$nin": ["encoder_opt"]},
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
            "context_encoder": "encoder",
            "encoder_opt": "encoder2d",
            "Ant_encoder": "encoder2d"
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
                if entry == "group" and config_entry in groups:
                    df[entry] = groups[config_entry]
                else:
                    df[entry] = config_entry
            if len(df) > 1:
                dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(fname, index=False)
    else:
        df = pd.read_csv(fname)
    return df


def main():
    fname = Path("data_context_gating_pendulum.csv")
    fname.parent.mkdir(parents=True, exist_ok=True)
    df = load_data_context_gating(fname)
    df = df[df["group"] != "hidden"]
    plt.tight_layout()
    ax = sns.lineplot(x="_step", y="eval/return", hue="group", data=df, ci=68)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean return over instances")
    ax.set_title("CARLPendulumEnv")
    ax.get_legend().set_title("Context")
    plt.savefig("context_gating_pendulum.png")
    plt.show()


if __name__ == "__main__":
    main()
