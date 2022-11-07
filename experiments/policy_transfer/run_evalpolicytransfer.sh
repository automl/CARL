#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_normal
#SBATCH --job-name=evalpolicytransfer
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M



python experiments/eval_policy_transfer.py
