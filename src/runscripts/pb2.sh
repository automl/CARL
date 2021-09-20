#!/bin/bash
#
#SBATCH --partition=cpu_normal
#SBATCH --job-name=pbt_bipedal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=80G

python run_pb2.py --seed $1
