#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_normal
#SBATCH --job-name=pb2eval
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --array=0-4

python experiments/play_pbt.py --env $1 --context_args $2 --policy_path $3 --name p$4 --seed $SLURM_ARRAY_TASK_ID

# visible
# sbatch runscripts/run_pb2.sh CARLAcrobotEnv link_length_1 /home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLAcrobotEnv/ray/pb2_mountaincar_gravity/pbt_policy_4943b_00000.txt 0
# hidden
# sbatch runscripts/run_pb2.sh CARLAcrobotEnv link_length_1 /home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLAcrobotEnv/ray/pb2_mountaincar_gravity_hidden/pbt_policy_52a07_00000.txt 0
# sbatch runscripts/run_pb2.sh CARLLunarLanderEnv GRAVITY_Y /home/eimer/Dokumente/git/meta-gym/src/results/experiments/pb2/CARLLunarLanderEnv/ray/pb2_ll_gravity_hidden/pbt_policy_b2e66_00000.txt

