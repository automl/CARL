# Benchmark RL Agents on CARL
Framework: coax.

## Runcommands
Local debugging
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/classic_control=pendulum' '+context_visibility=hidden' 'wandb.debug=true'
```

Example for slurm
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/classic_control=pendulum' '+context_visibility=hidden' 'wandb.debug=true' 'seed=range(1,11)' '+slurm=cpushort' -m
```

## Command Pieces

### Context Feature Args
Pendulum
```bash
'context_sampler.context_feature_names=[],[m, l, g, dt, max_speed],[m],[g],[l],[dt],[max_speed]'
```

### Context Sampling
Relative standard deviation
```bash
'context_sampler.sigma_rel=0.1,0.25,0.5'
```

### Context Visibility
```bash
'+context_visibility=hidden,cgate,visible_all,visible_changing'
```


## Experiment Plan
- [ ] SAC hidden, visible full, visible change only, SAC cGate, on CARLPendulum with 0.1, 0.25, 0.5 for all single context features + all context features at once, 100k
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=pendulum' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[m, l, g, dt, max_speed],[m],[g],[l],[dt],[max_speed]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpushort' 'hydra.launcher.cpus_per_task=1' 'hydra.launcher.timeout_min=240' -m
```
seeds [1-5]
`/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-08-07/12-00-34_benchmark_train`
seeds [6-10]
`/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-08-07/12-02-27_benchmark_train`


```bash
# Compare n_samples
python experiments/benchmarking/run_training.py 'experiment=benchmarking_n_samples' '+environments/classic_control=pendulum' 'seed=range(1,6)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm' 'context_sampler.context_feature_names=[],[m, l, g, dt, max_speed],[m],[g],[l],[dt],[max_speed]' 'context_sampler.n_samples=100,1000,10000' 'context_sampler.sigma_rel=0.1' '+slurm=cpushort' 'hydra.launcher.timeout_min=240' -m
```
280 combinations

- [ ] SAC hidden, SAC cGate on LunarLander planet contexts, 200k
```bash
'environments/box2d=lunarlander' '+context_visibility=hidden,cgate_hadamard'
```
- [ ] SAC hidden, SAC cGate on walker 0.1 for all context features + all context features at once, 1M
```bash
# Full
python experiments/benchmarking/run_training.py '+environments/dmc=walker' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping],[gravity],[density],[joint_damping]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m

# Only 0.1
python experiments/benchmarking/run_training.py '+environments/dmc=walker' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping],[gravity],[density],[joint_damping]' 'context_sampler.sigma_rel=0.1' '+slurm=cpu' -m
```
- [ ] SAC hidden, SAC cGate on quadruped 0.1 for all context features + whichever combination is manageable, 1M

### DMC Fish
- [ ] SAC hidden, SAC cGate on fish 0.1 for all context features + all context features at once, 500k
```bash
# Full
python experiments/benchmarking/run_training.py '+environments/dmc=fish' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping],[gravity],[density],[joint_damping]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m
# Half seeds
python experiments/benchmarking/run_training.py '+environments/dmc=fish' 'seed=range(1,6)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping],[gravity],[density],[joint_damping]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m
# Half seeds, hidden + cGate Hadamard
python experiments/benchmarking/run_training.py '+environments/dmc=fish' 'seed=range(1,6)' '+context_visibility=hidden,cgate_hadamard' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping],[gravity],[density],[joint_damping]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m
# Half seeds + cpu
python experiments/benchmarking/run_training.py '+environments/dmc=fish' 'seed=range(1,6)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping],[gravity],[density],[joint_damping]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpu' -m

# Half seeds + cpu +  only hidden + only [] + learning rates + network widths
python experiments/benchmarking/run_training.py '+environments/dmc=fish' 'seed=range(1,6)' '+context_visibility=hidden' 'context_sampler.context_feature_names=[]' 'context_sampler.sigma_rel=0.1' '+slurm=cpu' 'learning_rate=0.001,0.02,0.005'  'network.width=32,64' 'hydra.launcher.mem_gb=10' -m
 ```
- [ ] SAC hidden, SAC cGate on CartPole with train and test dists for the Kirk experiment (gravity + pole_length), 100k
- [ ] SAC hidden, SAC cGate Mario with only one context dist, 1M
- [ ] SAC hidden, SAC cGate brax HalfCheetah, 1M

### DMC Finger
gravity, limb_length_0, limb_length_1, spinner_radius, spinner_length
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/dmc=finger' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[limb_length_0,limb_length_1,spinner_length],[limb_length_0],[limb_length_1],[spinner_length]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m

# Debug
python experiments/benchmarking/training.py '+environments/dmc=finger' 'seed=4' '+context_visibility=cgate_hadamard' 'context_sampler.context_feature_names=[limb_length_0,limb_length_1,spinner_length]' 'context_sampler.sigma_rel=0.5' wandb.debug=true
```

### Ant
SAC
```bash
python experiments/benchmarking/run_training.py '+environments/brax=ant' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,friction,joint_stiffness],[gravity],[friction],[joint_stiffness]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m
```

- [ ] LSTM baseline for whatever makes sense

- MountainCar +  C51, 0.1, 0.25, 0.5, all visibilities, 100k steps, cfs: goal_velocity,force,gravity
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[goal_velocity,force,gravity],[goal_velocity],[force],[gravity]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpushort' -m

# Debug
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' max_num_frames=150_000 'seed=1' '+context_visibility=hidden' 'context_sampler.context_feature_names=[]' 'context_sampler.sigma_rel=0.1' 'wandb.debug=True'

# HPO
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' max_num_frames=1_000 'seed=456' '+context_visibility=hidden' 'context_sampler.context_feature_names=[]' '+hpo=mountaincar' 'wandb.debug=True' 'hydra.launcher=submitit_smac_local' -m
```

Acrobot
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=acrobot' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[link_length_1,link_length_2,link_mass_1,link_mass_2],[link_length_1],[link_length_2],[link_mass_1],[link_mass_2]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpushort' 'hydra.launcher.timeout_min=240' -m
```

```bash
# Debug
 python experiments/benchmarking/run_training.py max_num_frames=100000 '+environments/classic_control=acrobot' '+context_visibility=hidden' 'wandb.debug=true' '+algorithm=c51' 'eval_episodes=5' 
```



## Experiment Preparations
Find HPs for
- [x] Pendulum (SAC)
```bash
python experiments/benchmarking/run_training.py max_num_frames=10000 '+environments/classic_control=pendulum' '+context_visibility=hidden' 'wandb.debug=true' 'eval_episodes=5' 
```
- [ ] LunarLander (c51)
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/box2d=lunarlander' '+context_visibility=hidden' 'wandb.debug=true' '+algorithm=c51' 'eval_episodes=5' 
```
- [ ] Walker 
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/dmc=walker' '+context_visibility=hidden' 'wandb.debug=true' 'eval_episodes=5' 
```
- [ ] Quadruped
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/dmc=quadruped' '+context_visibility=hidden' 'wandb.debug=true' 'eval_episodes=5' 
```
- [ ] Fish
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/dmc=fish' '+context_visibility=hidden' 'wandb.debug=true' 'eval_episodes=5' 
```
- [ ] Mario c51 + conv encoder
```bash
 python experiments/benchmarking/run_training.py max_num_frames=25000 '+environments/misc=mario' '+context_visibility=hidden' 'wandb.debug=true' 'eval_episodes=5' 
```

## Evaluation Protocol [Kirk]
Debug
```bash
python experiments/benchmarking/run_training.py '+experiments=kirk_evaluation_protocol' 'wandb.debug=true' 'eval_episodes=5' 
```

Run
```bash
python experiments/benchmarking/run_training.py '+experiments=kirk_evaluation_protocol' 'seed=range(1,11)' 'kirk_evaluation_protocol.mode=A,B,C' '+context_visibility=hidden,cgate,visible_all,visible_changing' -m
```

Evaluate debug
```bash
python experiments/evaluation/evaluate.py +experiments=kirk_evaluation_protocol kirk_evaluation_protocol.distribution_type=test_extrapolation_all wandb.debug=True
```

```bash
python experiments/evaluation/run_evaluation.py '+experiments=kirk_evaluation_protocol' 'kirk_evaluation_protocol.distribution_type=train,test_interpolation,test_interpolation_combinatorial,test_extrapolation_single,test_extrapolation_all' -m --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-01/14-14-50_benchmark_train/'
```

## Landing in Space
```bash
# Debug
python experiments/benchmarking/training.py '+experiments=landing_in_space' 'wandb.debug=true' 'eval_episodes=5'

# Training
python experiments/benchmarking/run_training.py '+experiments=landing_in_space' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm' -m
python experiments/benchmarking/run_training.py '+experiments=landing_in_space' 'seed=range(1,11)' '+context_visibility=visible_all,visible_changing' -m  

# Result dirs
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-12/12-45-57_benchmark_train/
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-12/10-27-33_benchmark_train/
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-11/17-45-22_benchmark_train/
# concat all
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-14/16-21-59_benchmark_train

# Evaluation Run
python experiments/evaluation/run_evaluation.py '+experiments=landing_in_space' --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-14/16-21-59_benchmark_train' -m

# Evaluation Debug
python experiments/evaluation/evaluate.py '+experiments=landing_in_space' results_path='/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-12/12-45-57_benchmark_train/0' 'wandb.debug=True'

```

