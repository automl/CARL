#!/bin/bash
#
#SBATCH --partition=cpu_normal
#SBATCH --job-name=pb2_ant
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M

python run_pb2.py --env $1 --hp_file /home/eimer/Dokumente/git/meta-gym/src/hyperparameter.yml
