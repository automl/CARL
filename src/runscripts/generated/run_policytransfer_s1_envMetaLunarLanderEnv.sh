#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal
#SBATCH --job-name=policytransfer_s1_envMetaLunarLanderEnv
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000M



python experiments/run_policy_transfer.py --outdir results/experiments/policytransfer/MetaLunarLanderEnv   --seed 1 --env MetaLunarLanderEnv
