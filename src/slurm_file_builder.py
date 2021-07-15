from slurmbuilder.slurmbuilder import SlurmBuilder
import src.envs as envs

######################################
job_name = "meta_box2d"
env = "MetaBipedalWalkerEnv"
default_sample_std_percentage = 0.1
hide_context = False
######################################

print(env)
hide_context_dir_str = "_hidecontext" if hide_context else ""
hide_context_cmd_str = "--hide_context" if hide_context else ""

mail_user = "benjamin@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
time = "48:00:00"
mem_per_cpu = "2000M"
basecommand = "python run_stablebaselines.py --num_contexts 100 --steps 1000000 --add_context_feature_names_to_logdir"
cpus_per_task = "1"

env_defaults = getattr(envs, f"{env}_defaults")
outdir = f"results/singlecontextfeature_{default_sample_std_percentage}{hide_context_dir_str}/box2d/{env}"
basecommand += f" --outdir {outdir}  --num_workers {cpus_per_task} --default_sample_std_percentage {default_sample_std_percentage} {hide_context_cmd_str}"
context_features = list(env_defaults.keys())

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
        {
            "name": "context_feature_args",
            "id": "cfargs",
            "values": ['None'] + list(env_defaults.keys())  # None trains only with the default context
        }
    ]
)
sbuilder.build_shfiles()
