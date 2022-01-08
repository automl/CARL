from slurmbuilder.slurmbuilder import SlurmBuilder
import carl.envs as envs

######################################
job_name = "poltra"
env = "CARLLunarLanderEnv"
context_file = "experiments/lunarLander_contexts_train_2intervals.json"
context_file = "experiments/lunarLander_contexts_train_Gaussian.json"
######################################

mail_user = "benjamin@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
partition = "gpu_normal"
time = "03:00:00"
mem_per_cpu = "1000M" if env != "CARLVehicleRacingEnv" else "16000M"
basecommand = 'python experiments/policy_transfer/run.py'
# outdir = f"results/experiments/policy_transfer/{env}"
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
    partition=partition,
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
