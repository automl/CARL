from experiments.common.plot.wandb_utils import load_wandb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Data for cgate sb3 HPO
    # Pendulum 0.1 dt
    project_name = "tnt/carl"
    df_fname = "data/cgate_sb3_hpo.csv"
    config_entries = [
        "seed",
        "agent.kwargs.learning_rate",
        "agent.kwargs.policy_kwargs.context_branch_width",
    ]
    metrics = ["rollout/ep_rew_mean"]
    timestamp = datetime(
        2022,
        3,
        30,
        hour=17,
        minute=28,
        second=0,
    ).timestamp()
    filters = {
        "state": "finished",
        # "user": "benjamc",
        # "user.username": "benjamc",
        "summary_metrics._timestamp": {"$gt": timestamp},
    }
    redownload = False

    data = load_wandb(
        project_name=project_name,
        df_fname=df_fname,
        filters=filters,
        redownload=redownload,
        metrics=metrics,
        config_entries=config_entries,
    )

    groups = data.groupby(
        [
            "agent.kwargs.learning_rate",
            "agent.kwargs.policy_kwargs.context_branch_width",
        ]
    )

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(111)
    for group_id, group_df in groups:
        ax = sns.lineplot(
            data=group_df,
            x="_step",
            y="rollout/ep_rew_mean",
            ax=ax,
            label=group_id,
        )
    fig.set_tight_layout(True)
    plt.show()
