# Experiments


## Installation
Run following command to install the necessary packages (in root of repo).
```bash
pip install -e .[box2d,brax,dm_control,mario,dev,docs]
pip install -r experiments/requirements_experiments.txt
```

## Running Experiments
- Optimality Gap: `notes/optimality_gap.md`
- Context Efficiency: `notes/context_efficiency.md`
- General benchmarking / compounding factors: `experiments/benchmarking/README.md`

Currently the default is sampling contexts from a uniform distribution.
For general benchmarking, the command is 
`python experiments/benchmarking/run_training.py '+environments/classic_control=pendulum' 'seed=range(1,11)' '+context_visibility=glob(*)' 'context_sampler.context_feature_names=[],[m, l, g, dt, max_speed],[m],[g],[l],[dt],[max_speed],[m,g],[m,g,l],[m,g,l,dt]' '+context_sampling=glob(*)' '+slurm=cpushort' 'hydra.launcher.cpus_per_task=1' 'hydra.launcher.timeout_min=240' -m`

The general benchmarking logs to wandb. Context efficiency and optimality gap do not and have a special folder structure.

## Analyzing / Evaluating Experiments
- Optimality Gap: `experiments/evaluation/eval_optimality_gap.ipynb`
- Context Efficiency: `experiments/evaluation/eval_context_efficiency.ipynb`
- General benchmarking: `experiments/evaluation/eval_benchmarking.ipynb` -- plots training performance

