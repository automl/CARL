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
# train
python experiments/benchmarking/run_training.py '+environments/brax=ant' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity,friction,joint_stiffness],[gravity],[friction],[joint_stiffness]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=gpu' -m

# debug on cluster
python experiments/benchmarking/run_training.py '+environments/brax=ant' 'seed=range(1,2)' '+context_visibility=cgate_hadamard' 'context_sampler.context_feature_names=[gravity,friction,joint_stiffness]' 'context_sampler.sigma_rel=0.5' '+slurm=gpu' 'wandb.debug=true' -m


# debug local
python experiments/benchmarking/run_training.py '+environments/brax=ant' 'seed=1' '+context_visibility=cgate_hadamard' 'context_sampler.context_feature_names=[gravity,friction,joint_stiffness]' 'context_sampler.sigma_rel=0.5' 'wandb.debug=true'

# train uniform benchmarking
python experiments/benchmarking/run_training.py '+environments/brax=ant' 'seed=range(1,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[gravity,friction,joint_stiffness]' '+context_sampling=glob(*)' '+slurm=gpu' -m
```

Halfcheetah
```bash
# train uniform benchmarking
python experiments/benchmarking/run_training.py '+environments/brax=halfcheetah' 'seed=range(1,6)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[gravity,friction,torso_mass]' '+context_sampling=glob(*)' '+slurm=gpu' -m
```

- [ ] LSTM baseline for whatever makes sense

- MountainCar +  C51, 0.1, 0.25, 0.5, all visibilities, 100k steps, cfs: goal_velocity,force,gravity
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[goal_velocity,force,gravity],[goal_velocity],[force],[gravity]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpushort' -m

# uniform
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' 'seed=range(1,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[goal_velocity,force,gravity],[goal_velocity],[force],[gravity]' '+context_sampling=glob(*)' '+slurm=cpushort' -m

# Debug
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' max_num_frames=150_000 'seed=1' '+context_visibility=hidden' 'context_sampler.context_feature_names=[]' 'context_sampler.sigma_rel=0.1' 'wandb.debug=True'

# HPO
python experiments/benchmarking/run_training.py '+environments/classic_control=mountaincar' max_num_frames=1_000 'seed=456' '+context_visibility=hidden' 'context_sampler.context_feature_names=[]' '+hpo=mountaincar' 'wandb.debug=True' 'hydra.launcher=submitit_smac_local' -m
```

Acrobot
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=acrobot' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[link_length_1,link_length_2,link_mass_1,link_mass_2],[link_length_1],[link_length_2],[link_mass_1],[link_mass_2]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpushort' 'hydra.launcher.timeout_min=240' -m

# uniform bounds
python experiments/benchmarking/run_training.py '+environments/classic_control=acrobot' 'seed=range(1,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[link_length_1,link_length_2,link_mass_1,link_mass_2]' '+context_sampling=glob(*)' '+slurm=cpushort' 'hydra.launcher.timeout_min=240' -m
```

```bash
# Debug
 python experiments/benchmarking/run_training.py max_num_frames=100000 '+environments/classic_control=acrobot' '+context_visibility=hidden' 'wandb.debug=true' '+algorithm=c51' 'eval_episodes=5' 
```

CartPole
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=cartpole' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[gravity],[masscart],[masspole],[pole_length],[gravity,masscart,masspole,pole_length]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpushort' 'hydra.launcher.timeout_min=240' -m
```

Bipedal Walker
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/box2d=bipedalwalker' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[FRICTION],[GRAVITY_Y],[MOTORS_TORQUE],[FRICTION,GRAVITY_Y,MOTORS_TORQUE]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpu' -m
```

Lunar Lander
- normal distribution
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/box2d=lunarlander' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[MAIN_ENGINE_POWER],[GRAVITY_Y],[SIDE_ENGINE_POWER],[MAIN_ENGINE_POWER,GRAVITY_Y,SIDE_ENGINE_POWER]' 'context_sampler.sigma_rel=0.1,0.25,0.5' '+slurm=cpu' -m
```

- uniform distribution
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/box2d=lunarlander' 'seed=range(1,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[MAIN_ENGINE_POWER],[GRAVITY_Y],[MAIN_ENGINE_POWER,GRAVITY_Y]' '+context_sampling=glob(*)' '+slurm=cpu' -m
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
experiment=kirk_evaluation_protocol2

Debug
```bash
python experiments/benchmarking/run_training.py '+experiments=kirk_evaluation_protocol' 'wandb.debug=true' 'eval_episodes=5' 
```

Run
```bash
python experiments/benchmarking/run_training.py '+experiments=kirk_evaluation_protocol' 'seed=range(1,11)' 'kirk_evaluation_protocol.mode=A,B,C' '+context_visibility=hidden,visible_all,cgate_hadamard,cgate_lstm' -m
```

Evaluate debug
```bash
python experiments/evaluation/evaluate.py +experiments=kirk_evaluation_protocol kirk_evaluation_protocol.distribution_type=test_extrapolation_all wandb.debug=True
```


```bash
# old
python experiments/evaluation/run_evaluation.py '+experiments=kirk_evaluation_protocol' 'kirk_evaluation_protocol.distribution_type=train,test_interpolation,test_interpolation_combinatorial,test_extrapolation_single,test_extrapolation_all' -m --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-01/14-14-50_benchmark_train/'

# new
python experiments/evaluation/run_evaluation.py '+experiments=kirk_evaluation_protocol' 'kirk_evaluation_protocol.distribution_type=train,test_interpolation,test_interpolation_combinatorial,test_extrapolation_single,test_extrapolation_all' -m --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-10-27/17-41-56_benchmark_train'

# rerun (21.12.2022)
python experiments/evaluation/run_evaluation.py '+experiments=kirk_evaluation_protocol' 'kirk_evaluation_protocol.distribution_type=train,test_interpolation,test_interpolation_combinatorial,test_extrapolation_single,test_extrapolation_all' -m --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-12-21/17-19-39_benchmark_train'
```

## Landing in Space
```bash
# Debug
python experiments/benchmarking/training.py '+experiments=landing_in_space' 'wandb.debug=true' 'eval_episodes=5'

# Training
python experiments/benchmarking/run_training.py '+experiments=landing_in_space' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm' -m
python experiments/benchmarking/run_training.py '+experiments=landing_in_space' 'seed=range(1,11)' '+context_visibility=visible_all,visible_changing' -m

python experiments/benchmarking/run_training.py '+experiments=landing_in_space' 'seed=range(1,11)' '+context_visibility=glob(*)' '+slurm=cpushort' -m    

# Result dirs
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-12/12-45-57_benchmark_train/
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-12/10-27-33_benchmark_train/
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-11/17-45-22_benchmark_train/
# concat all
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-14/16-21-59_benchmark_train

# current
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-10-14/11-50-33_benchmark_train/
# /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-10-19/12-27-11_benchmark_train

# Evaluation Run
python experiments/evaluation/run_evaluation.py '+experiments=landing_in_space' --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-10-19/12-27-11_benchmark_train' -m

# Evaluation Debug
python experiments/evaluation/evaluate.py '+experiments=landing_in_space' results_path='/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-12/12-45-57_benchmark_train/0' 'wandb.debug=True'

```


### New
## Acrobot
```bash
# Train
python experiments/benchmarking/run_training.py '+environments/classic_control=acrobot' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,cgate_lstm,visible_all,visible_changing' 'context_sampler.context_feature_names=[],[link_length_1,link_length_2,link_mass_1,link_mass_2],[link_length_1],[link_length_2],[link_mass_1],[link_mass_2]' '+context_sampling=glob(*)' '+slurm=cpushort' 'hydra.launcher.timeout_min=240' -m
```

## Walker
```bash
# ,[gravity],[density],[joint_damping]
python experiments/benchmarking/run_training.py '+environments/dmc=walker' 'seed=range(1,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[gravity,density,joint_damping]' '+slurm=gpu' '+context_sampling=glob(*)' -m
```

## Optimality Gap
1. Train general agent
```bash
# General Agent
python experiments/benchmarking/run_training.py 'seed=range(1,11)' '+experiments=optimality_gap' -m
```
2. Evaluate general agent
3. Prepare context files and runscripts with `prepare_optimality_gap_oracle_training.ipynb`
4. Train oracles with `for i in {0..9}; do bash experiments/evaluation/tmp/runcommands_optimality_gap/run_$i.sh; done`
5. Evaluate oracles with 
6. Collect run folders of general agent and oracles and use `eval_optimality_gap.ipynb` for plotting


Results: optimality_gap3
general agent dir: `/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-13/10-09-12_benchmark_train`
eval dir general agent: `/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/multirun/2022-11-13/11-09-22`
train dirs optimal agents: 
```python
train_dirs = [

]
eval_dirs = [

]
```




## Context Sample Efficiency
Pendulum, l, sigma rel = 0.25, hidden vs visible, 
n_contexts=1,2,4,8,16,32,64,128,256,512,1024
10 seeds

```bash
python experiments/benchmarking/run_training.py 'seed=range(1,11)' '+experiments=context_efficiency' 'context_sampler.n_samples=1,2,4,8,16,32,64,128,256,512,1024' '+context_visibility=hidden,visible_changing' -m
```

```python
# hidden, visible_changing
result_dir = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-02/21-22-36_benchmark_train"
# cgate_hadamard, cgate_lstm, visible_all
result_dir = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-03/17-15-06_benchmark_train"
```

Create eval contexts with `experiments/notebooks/create_context_efficiency_test_contexts.ipynb`.

Evaluation
```bash
# hidden, visible_changing
python experiments/evaluation/run_evaluation.py 'contexts_path=/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/experiments/evaluation/data/context_efficiency/contexts_evaluation.json' 'n_eval_episodes_per_context=10' --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-02/21-22-36_benchmark_train' -m

# eval result dir:  "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/multirun/2022-11-03/12-29-52/"

# cgate_hadamard, cgate_lstm, visible_all
python experiments/evaluation/run_evaluation.py 'contexts_path=/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/experiments/evaluation/data/context_efficiency/contexts_evaluation.json' 'n_eval_episodes_per_context=10' --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-03/17-15-06_benchmark_train' -m

# eval result dir: "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/multirun/2022-11-03/19-59-39"
```

The whole thing with a uniform distribution:
```bash
python experiments/benchmarking/run_training.py 'seed=range(1,11)' '+experiments=context_efficiency' 'context_sampler.n_samples=1,2,4,8,16,32,64,128' '+context_visibility=hidden,visible_changing,visible_all,cgate_hadamard,cgate_lstm' -m
```

Result dir = `/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-07/14-01-36_benchmark_train`
Rerun (only cgate_lstm) = `/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-13/10-59-12_benchmark_train`



Eval:
```bash
python experiments/evaluation/run_evaluation.py 'contexts_path=/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/experiments/evaluation/data/context_efficiency/contexts_evaluation_1024.json' 'n_eval_episodes_per_context=10' --result_dir '/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-07/14-01-36_benchmark_train' -m

# result dir: /home/benjamin/Dokumente/code/tmp/tntcomp/CARL/multirun/2022-11-07/20-45-00
```



 Rerun with SAC, context_efficiency2, hadamard small network width
 result dir: `/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-11-13/11-06-36_benchmark_train`

 ## Compounding
 ```bash
 python experiments/benchmarking/run_training.py '+environments/classic_control=pendulum' 'seed=range(6,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[m, l, g, dt, max_speed],[m],[g],[l],[dt],[max_speed],[m,l],[m,l,g],[m,l,g,dt]' '+context_sampling=glob(*)' '+slurm=cpu' 'hydra.launcher.cpus_per_task=1' 'hydra.launcher.timeout_min=240' -m
 ```



 # Check
 ```bash
 python experiments/benchmarking/run_training.py '+environments/classic_control=pendulum' 'seed=range(1,11)' '+context_visibility=hidden,cgate_hadamard,visible_changing' 'context_sampler.context_feature_names=[],[m],[g],[l],[dt],[max_speed]' '+context_sampling=glob(*)' '+slurm=cpu' 'hydra.launcher.cpus_per_task=1' 'hydra.launcher.timeout_min=240' 'wandb.tags=[rerun]' -m
 ```

 ## New Bounds For Pendulum
 g, l, m, [1, 2.2]
 ```bash
  python experiments/benchmarking/run_training.py '+environments/classic_control=pendulum' 'seed=range(1,11)' '+context_visibility=glob(*)' 'experiment=benchmarking_u_wider' '+context_sampling=uniform_10' '+slurm=cpushort' 'hydra.launcher.cpus_per_task=1' 'hydra.launcher.timeout_min=240' 'context_sampler.context_feature_names=[],[g],[l],[m]' 'context_sampler.uniform_bounds_rel=[1, 2.2]' -m
 ```

Acrobot
 link_length_1
 link_mass_2
 ```bash
  python experiments/benchmarking/run_training.py '+environments/classic_control=acrobot' 'seed=range(1,11)' '+context_visibility=glob(*)' 'experiment=benchmarking_u_wider' '+context_sampling=uniform_10' '+slurm=cpushort' 'hydra.launcher.cpus_per_task=1' 'hydra.launcher.timeout_min=240' 'context_sampler.context_feature_names=[],[link_length_1],[link_mass_2]' 'context_sampler.uniform_bounds_rel=[1, 2.5]' -m
 ```