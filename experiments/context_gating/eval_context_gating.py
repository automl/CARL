import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from src.eval.plotting_style import set_rc_params
from tqdm import tqdm


def main():
    api = wandb.Api()
    runs = api.runs(
        "tnt/carl",
        filters={
            "config.carl.gaussian_noise_std_percentage": {"$in": [0.4]},
            "config.carl.scale_context_features": "no",
            "state": "finished",
        },
    )
    dfs = []
    config_entries = [
        "group",
        "contexts.context_feature_args",
        "carl.state_context_features",
        "carl.gaussian_noise_std_percentage",
    ]
    metrics = ["eval/return"]
    groups = {
        "hidden_context": "hidden",
        "concat_context": "concat",
        "context_gating": "gating",
    }
    runs = list(runs)
    for run in tqdm(runs):
        df = pd.DataFrame()
        for i, row in run.history(keys=metrics).iterrows():
            if all([metric in row for metric in metrics]):
                df = df.append(row, ignore_index=True)
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
    set_rc_params()
    plt.tight_layout()
    ax = sns.lineplot(x="_step", y="eval/return", hue="group", data=df, ci=68)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean return over instances")
    ax.set_title("CARLPendulumEnv")
    ax.get_legend().set_title("Context")
    plt.savefig("context_gating_pendulum.png")


if __name__ == "__main__":
    main()
