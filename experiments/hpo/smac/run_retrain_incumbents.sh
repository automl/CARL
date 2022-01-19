#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_normal
#SBATCH --job-name=CARLreinc
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --array=0-34
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M


STEPS=1000000
python training/retrain_incumbents.py --expdir /home/benjamin/Dokumente/code/tmp/carl/src/results/optimized/classic_control/CARLCartPoleEnv/0.1_contexthidden --steps $STEPS --incumbent_idx $SLURM_ARRAY_TASK_ID
