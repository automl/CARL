import os
import glob
from slurmbuilder.slurmbuilder import SlurmBuilder
import src.envs as envs

######################################
job_name = "CARL"
env = "CARLCartPoleEnv"
envtype = "classic_control"
default_sample_std_percentage = 0.1
hide_context = True
vec_env_cls = "DummyVecEnv"
agent = "PPO"
n_timesteps = 1000000
state_context_features = "changing_context_features"
no_eval = True
hp_opt = True
on_luis = True
branch_name = "HP_opt"
######################################

runfile = "train.py" if not hp_opt else "training/hp_opt.py"
exptype = "base_vs_context" if not hp_opt else "optimized"

print(env)
hide_context_dir_str = "_hidecontext" if hide_context else ""
hide_context_cmd_str = "--hide_context" if hide_context else ""
eval_freq = 10000 if "Racing" in env else 5000
xvfb_str = "" if not "Racing" in env else "xvfb-run "
eval_freq = 5000 if envtype == "classic_control" else eval_freq
eval_freq = 50000
eval_cb = "" if not no_eval else " --no_eval_callback"

partition = "gpu_normal" if not on_luis else "amo"
gres = "gpu:1"
mail_user = "benjamin@tnt.uni-hannover.de" if not on_luis else "benjamins@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
time = "72:00:00"
mem_per_cpu = "2000M" if "racing" not in env else "8000M"
basecommand = f"cd src\n{xvfb_str}python {runfile} --num_contexts 100 --steps {n_timesteps} " \
              f"--add_context_feature_names_to_logdir --hp_file training/hyperparameters/hyperparameters_ppo.yml"
cpus_per_task = "8"
pre_command = ""
post_command = ""
runcommands_file_precommand = ""
if on_luis:
    luis_template_fn = "utils/LUIS_template.sh"
    conda_env_name = "carl"
    project_name = "CARL"
    with open(luis_template_fn, 'r') as file:
        content = file.read()
    pre_command = content.format(conda_env_name=conda_env_name, project_name=project_name, branch_name=branch_name)
    runcommands_file_precommand = "git pull\n"
    output_filename = os.path.join("$WORKING_DIR", output_filename)

    # remove everything in runscripts/generated
    runfile_dir = "runscripts/generated"
    files = glob.glob(os.path.join(runfile_dir, "*.sh"))
    for file in files:
        os.remove(file)

env_defaults = getattr(envs, f"{env}_defaults")
outdir = f"results/singlecontextfeature_{default_sample_std_percentage}{hide_context_dir_str}/{envtype}/{env}"
hide_context_dir_str = "contexthidden" if hide_context else "contextvisible"
state_context_features_str = "changing" if state_context_features is not None else ""
outdir = f"results/{exptype}/{envtype}"
if on_luis:
    outdir = os.path.join("$WORKING_DIR", outdir)
basecommand += f" --outdir {outdir}  --num_workers {cpus_per_task} --build_outdir_from_args"
basecommand += f" {hide_context_cmd_str} --eval_freq {eval_freq} --seed $SLURM_ARRAY_TASK_ID " \
               f"--scale_context_features no  --vec_env_cls {vec_env_cls}  --agent {agent} "
basecommand += eval_cb
if state_context_features is not None:
    basecommand += f"--state_context_features {state_context_features}"
context_features = list(env_defaults.keys())
seeds = [0, 1, 2, 3, 4]
array = f"{min(seeds)}-{max(seeds)}"

slurm_config = {
        "mail-user": mail_user,
        "mail-type": 'ALL',
        "partition": partition,
        "job-name": job_name,
        "output": output_filename,
        "array": array,
        "time": time,
        # "nodes": nodes,
        # "tasks-per-node": tasks_per_node,
        # "ntasks": ntasks,
        # "mincpus": mincpus,
        "cpus-per-task": cpus_per_task,
        "mem-per-cpu": mem_per_cpu,
        "gres": gres,
    }
if on_luis and partition == "amo":
    del slurm_config["gres"]

sbuilder = SlurmBuilder(
    slurm_config=slurm_config,
    pre_command=pre_command,
    base_command=basecommand,
    post_command=post_command,
    output_filename=output_filename,
    runcommands_file_precommand=runcommands_file_precommand,
    iteration_list=[
        {
            "name": "env",
            "id": "env",
            "values": [env]
        },
        {
            "name": "default_sample_std_percentage",
            "id": "std",
            "values": [0.1, 0.25, 0.5],
        },
        {
            "name": "hide_context",
            "id": "vis",
            "values": [True, False]
        },
        {
            "name": "context_feature_args",
            "id": "cfargs",
            "values": ['None'] + list(env_defaults.keys())  # None trains only with the default context
        }
    ]
)
sbuilder.build_shfiles()

transfer_fn = "runscripts/transfer_generated.sh"
content = "scp -r runscripts/generated/  nhmlbenc@transfer.cluster.uni-hannover.de:/home/nhmlbenc/repos/CARL/src/runscripts/"
with open(transfer_fn, 'w') as file:
    file.write(content)
