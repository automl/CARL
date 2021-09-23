#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal
#SBATCH --job-name=meta_box2d_envMetaBipedalWalkerEnv_cfargsSCALE
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --array=0-4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M



python run_stablebaselines.py --num_contexts 100 --steps 1000000 --add_context_feature_names_to_logdir --hp_file hyperparameter.yml --outdir results/singlecontextfeature_0.0_hidecontext/box2d/MetaBipedalWalkerEnv  --num_workers 1 --default_sample_std_percentage 0.0 --hide_context --eval_freq 1000 --seed $SLURM_ARRAY_TASK_ID --scale_context_features no --env MetaBipedalWalkerEnv --context_feature_args SCALE
