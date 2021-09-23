#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal
#SBATCH --job-name=meta_box2d_s1_envMetaLunarLanderEnv_LEG_AWAY
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M



python run_stablebaselines.py --num_contexts 100 --steps 1000000 --outdir results/box2d/MetaLunarLanderEnv  --num_workers 1 --seed 1 --env MetaLunarLanderEnv -- LEG_AWAY
