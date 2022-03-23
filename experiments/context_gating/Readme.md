# Context Gating Experiments

To reproduce the experiments use the following command:

```bash
python run_context_gating.py 'seed=range(0,5)' '+experiment=glob(*)' '+algorithm=td3' '+environment=pendulum' 'contexts.context_feature_args'contexts.context_feature_args=[],[g],[max_speed],[l],[m],[dt]' 'carl.state_context_features=null,${contexts.context_feature_args}' 'contexts.default_sample_std_percentage=0.1,0.25,0.5' --multirun
```

## Hyperparameter Sweep

```bash
python run_context_gating.py '+experiment=context_gating' '+algorithm=sac' '+environment=pendulum' '+sweeps=sac_hp_search' --multirun
```

Please note that `glob` does not work. The experiments need to be started and swept over individually.
