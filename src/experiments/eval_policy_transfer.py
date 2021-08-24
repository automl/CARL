import os
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

from src.eval.eval_models import load_model, setup_env
from src.experiments.policy_transfer import get_train_contexts_ll, gravities
from src.context_sampler import get_default_context_and_bounds

envname = "MetaLunarLanderEnv"
outdir = "results/experiments/policytransfer"
outdir = os.path.join(outdir, envname)
eval_data_fname = Path(outdir) / "eval_data.csv"
collect_data = False if eval_data_fname.is_file() else True
collect_data = True

# create test contexts (train distribution)
context_feature_key = "GRAVITY_Y"
n_contexts = 100
n_eval_eps = n_contexts
env_default_context, env_bounds = get_default_context_and_bounds(env_name=envname)
contexts_test = get_train_contexts_ll(gravities, context_feature_key, n_contexts, env_default_context)

planets_test_in = ["Mars", "Moon"]
planets_test_out = ["Jupiter", "Neptune", "Pluto"]

if collect_data:
    data = []
    model_fnames = glob.glob(os.path.join(outdir, "*", "model.zip"))
    for model_fname in model_fnames:
        model_fname = Path(model_fname)
        print(model_fname)
        model, info = load_model(model_fname)

        # test contexts (train distribution)
        context_feature_ids = list(gravities.keys())
        test_setup = {
            "train_distribution": contexts_test
        }
        for context_feature_id in context_feature_ids:
            context = env_default_context.copy()
            context[context_feature_key] = gravities[context_feature_id]
            contexts = {0: context}
            test_setup[context_feature_id] = contexts

        for context_feature_id, contexts in test_setup.items():
            env = setup_env(model_fname.parent, contexts=contexts)

            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=n_eval_eps,
                return_episode_rewards=True
            )
            D = pd.DataFrame({
                "planet": [context_feature_id] * n_eval_eps,
                "ep_rew_mean": mean_reward,
                "ep_rew_std": std_reward,
                "train_seed": [info['seed']] * n_eval_eps,
            })
            data.append(D)
    data = pd.concat(data)
    data.to_csv(eval_data_fname, sep=";")

data = pd.read_csv(eval_data_fname, sep=";")
figsize = (6, 4)
dpi = 200
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_subplot(111)
ax = sns.boxplot(data=data, x="planet", y="ep_rew_mean", ax=ax)
ax.set_ylim(-10000, 0)
plt.show()




