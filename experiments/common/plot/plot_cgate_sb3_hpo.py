from experiments.common.plot.wandb_utils import load_wandb
from datetime import datetime

if __name__ == '__main__':
    # Data for cgate sb3 HPO
    # Pendulum 0.1 dt
    project_name = "tnt/carl"
    df_fname = "data/cgate_sb3_hpo.csv"
    config_entries = [
        "seed"
        "agent.kwargs.learning_rate",
        "agent.kwargs.policy_kwargs.context_branch_width"
    ]
    metrics = ["rollout/ep_rew_mean"]
    timestamp = datetime(2022, 3, 30, hour=17, minute=28, second=0,).timestamp()
    filters = {
        "state": "finished",
        # "user": "benjamc",
        # "user.username": "benjamc",
        "summary_metrics._timestamp": {"$gt": timestamp}
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