# Policy Transfer Experiment
To rerun the LunarLander policy transfer experiment, we recommend you use slurm in combination with the 'spawner_policy-transfer.py' script to automatically create all the runscripts you need.
Alternatively, you can first manually sample contexts:
```bash
python sample_lunarlander_contexts.py 
```
Afterwards, run:
```bash
python run.py --seed 0 --env CARLLunarLanderEnv --context_file experiments/lunarLander_contexts_train_Gaussian.json
```
