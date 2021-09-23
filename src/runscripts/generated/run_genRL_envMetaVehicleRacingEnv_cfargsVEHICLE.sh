#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal
#SBATCH --job-name=genRL_envMetaVehicleRacingEnv_cfargsVEHICLE
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --array=0-4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M



xvfb-run python run_stablebaselines.py --num_contexts 100 --steps 1000000 --add_context_feature_names_to_logdir --hp_file hyperparameter.yml --outdir results/singlecontextfeature_0.1_hidecontext/box2d/MetaVehicleRacingEnv  --num_workers 1 --default_sample_std_percentage 0.1 --hide_context --eval_freq 10000 --seed $SLURM_ARRAY_TASK_ID --scale_context_features no --env MetaVehicleRacingEnv --context_feature_args VEHICLE
