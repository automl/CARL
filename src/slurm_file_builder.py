from slurmbuilder.slurmbuilder import SlurmBuilder
import src.envs as envs

mail_user = "eimer@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
time = "48:00:00"
mem_per_cpu = "2000M"
basecommands = {
        "classic_control": "python run_stablebaselines.py --num_contexts 100 --steps 1000000"
    }
cpus_per_task = "1"
env = "MetaAnt"#"MetaMountainCarEnv", "MetaAcrobotEnv", "MetaCartPoleEnv", "MetaPendulumEnv"
env_defaults = getattr(envs, f"{env}_defaults")
outdir = f"results/brax/{env}"
for job_name, basecommand in basecommands.items():
    basecommand += f" --outdir {outdir}  --num_workers {cpus_per_task}"

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
                "name": "",
                "id": "",
                "values": list(env_defaults.keys())
            }
        ]
    )
    sbuilder.build_shfiles()
