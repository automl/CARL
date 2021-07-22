from slurmbuilder.slurmbuilder import SlurmBuilder
import src.envs as envs

######################################
job_name = "policytransfer"
env = "MetaLunarLanderEnv"
######################################

mail_user = "benjamin@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
time = "24:00:00"
mem_per_cpu = "1000M" if env != "MetaVehicleRacingEnv" else "16000M"
basecommand = 'python experiments/run_policy_transfer.py'
outdir = f"results/experiments/policytransfer/{env}"
basecommand += f" --outdir {outdir}  "
cpus_per_task = "1"


sbuilder = SlurmBuilder(
    job_name=job_name,
    mail_user=mail_user,
    base_command=basecommand,
    time=time,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    output_filename=output_filename,
    iteration_list=[
        {
            "name": "seed",
            "id": "s",
            "values": [0, 1, 2, 3, 4],
        },
        {
            "name": "env",
            "id": "env",
            "values": [env]
        },
    ]
)
sbuilder.build_shfiles()
