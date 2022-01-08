# Context Gating Experiments

To reproduce the experiments use the following command:

```bash
python run_context_gating.py 'seed=range(0,5)' '+experiment=glob(*)' '+algorithm=sac' '+environment=pendulum' 'contexts.context_feature_args=[],[g],[max_speed],[l],[m],[dt]' 'carl.state_context_features=null,${contexts.context_feature_args}' 'carl.gaussian_noise_std_percentage=0.4' --multirun
```

## Hyperparameter Sweep

```bash
python run_context_gating.py '+experiment=glob(*)' '+algorithm=sac' '+environment=pendulum' '+sweeps=sac_hp_search' --multirun
```
