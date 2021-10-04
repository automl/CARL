#!/bin/bash
#
#SBATCH --partition=gpu_normal
#SBATCH --job-name=pb2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=80G

python experiments/run_pb2.py --seed 0
