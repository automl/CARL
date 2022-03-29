# Train 
Command for `train_sb3.py` which integrates hydra and wandb.

```bash
python experiments/common/train/train_sb3.py 'debug=true' '+environment=pendulum' '+agent=ddpg' '+experiment=cgate'
```


Debug
```
python experiments/common/train/train_sb3.py 'wandb.debug=true' '+environment=pendulum' '+agent=ddpg' '+experiment=cgate' 'wandb.job_type=debug' 'wandb.notes="this is a test"'
```


