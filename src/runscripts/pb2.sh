#!/bin/bash
#
#SBATCH --partition=cpu_normal
#SBATCH --job-name=pb2_ant
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000M
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal
#SBATCH --output=slurmout/slurm-%j.out

python run_pb2.py --seed $1
