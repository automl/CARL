import pandas as pd
from pathlib import Path
from experiments.common.eval.gather_data import collect_results

paths = [
    "results/base_vs_context/classic_control/CARLPendulumEnv/0.5_contextvisible",  # DDPG
]

data = []
multiindex = []
for path in paths:
    path = Path(path)
    if path.is_dir():
        results = collect_results(path)
        counter = 0
        pathidx = "/".join(Path(path).parts[-1:])

        if results:
            for cf_name, df in results.items():
                n_seeds = df["seed"].nunique()
                seeds = [g[0] for g in df.groupby("seed")]
                n_points_per_seed = [len(g[1]) for g in df.groupby("seed")]
                n_points = len(df)
                n_avg_per_seed = n_points / n_seeds
                data.append({
                    "context_feature_name": cf_name,
                    "n_seeds": n_seeds,
                    "n_points": n_points,
                    "n_avg_per_seed": n_avg_per_seed,
                    "seeds": seeds,
                    "n_points_per_seed": n_points_per_seed,
                    "path": path,
                })
                counter += 1
                multiindex.append((pathidx, counter))
        else:
            data.append({
                "context_feature_name": 0,
                "n_seeds": 0,
                "n_points": 0,
                "n_avg_per_seed": 0,
                "seeds": [],
                "n_points_per_seed": 0,
                "path": path,
            })
            counter += 1
            multiindex.append((pathidx, counter))

index = pd.MultiIndex.from_tuples(multiindex, names=["path", "idx"])
data = pd.DataFrame(data, index=index)
