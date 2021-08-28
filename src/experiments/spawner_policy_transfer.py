from slurmbuilder.slurmbuilder import SlurmBuilder
import src.envs as envs

######################################
job_name = "poltra"
env = "CARLLunarLanderEnv"
context_file = "results/experiments/policytransfer/new/CARLLunarLanderEnv/hidden/GRAVITY_Y/contexts_train.json"
######################################

mail_user = "benjamin@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
time = "24:00:00"
mem_per_cpu = "1000M" if env != "CARLVehicleRacingEnv" else "16000M"
basecommand = 'python experiments/run_policy_transfer.py'
# outdir = f"results/experiments/policytransfer/{env}"
# basecommand += f" --outdir {outdir}  "
basecommand += " --seed $SLURM_ARRAY_TASK_ID "
basecommand += f" --context_file {context_file}"
cpus_per_task = "1"

seeds = [0, 1, 2, 3, 4]
array = f"{min(seeds)}-{max(seeds)}"


sbuilder = SlurmBuilder(
    job_name=job_name,
    mail_user=mail_user,
    base_command=basecommand,
    time=time,
    array=array,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    output_filename=output_filename,
    iteration_list=[
        {
            "name": "env",
            "id": "env",
            "values": [env]
        },
    ]
)
sbuilder.build_shfiles()
