import os
import glob
import numpy as np
from slurmbuilder.slurmbuilder import SlurmBuilder
import carl.envs as envs
from pathlib import Path
cwd = os.getcwd()
if "runscripts" in cwd:
    os.chdir(Path(cwd).parent)

#########################################################################
job_name = "CARL"
env = "CARLPendulumEnv"
envtype = "classic_control"
default_sample_std_percentage = 0.1
hide_context = True
vec_env_cls = "DummyVecEnv"
agent = "DDPG"
n_timesteps = 1_000_000
state_context_features = "changing_context_features"
no_eval = False
hp_opt = False
use_cpu = True
on_luis = False
use_cgate = True
follow_evaluation_protocol = False
luis_user_name = "nhmlbenc"  # can be empty string if not on LUIS
branch_name = "HP_opt"
time = "12:00:00" if use_cpu else "24:00:00"
tnt_cpu_partition = "short"
outdirbase = "results/cGate2"
if follow_evaluation_protocol:
    outdirbase = "results/evaluation_protocol"
#########################################################################
context_file = "envs/box2d/parking_garage/context_set_all.json"  # only relevant for vehicle racing env
compounding_pendulum_cfargs = ['None', 'm', 'm l', 'm l dt', 'm l dt g', 'm l dt g max_speed']
env_defaults = getattr(envs, f"{env}_defaults")
iteration_list = [
        {
            "name": "env",
            "id": "env",
            "values": [env]
        },
        {
            "name": "default_sample_std_percentage",
            "id": "std",
            "values": [0.1, 0.25, 0.5]
        },
        {
            "name": "hide_context",
            "id": "hid",
            "values": [True, False]
        },
        {
            "name": "context_feature_args",
            "id": "cfargs",
            "values": ['None'] + list(env_defaults.keys())  # None trains only with the default context
        }
    ]
#########################################################################
if follow_evaluation_protocol:
    iteration_list = [
        {
            "name": "env",
            "id": "env",
            "values": [env]
        },
        {
            "name": "follow_evaluation_protocol",
            "id": "ep",
            "values": [True]
        },
        {
            "name": "hide_context",
            "id": "hid",
            "values": [False, True]
        },
        {
            "name": "evaluation_protocol_mode",
            "id": "mode",
            "values": ['A', 'B', 'C']
        }
    ]

runfile = "experiments/common/train/train.py" if not hp_opt else "training/hp_opt.py"
exptype = "base_vs_context" if not hp_opt else "optimized"

print(env)
hide_context_dir_str = "_hidecontext" if hide_context else ""
hide_context_cmd_str = "--hide_context" if hide_context else ""
eval_freq = 10000 if "Racing" in env else 5000
use_xvfb = True if "Racing" in env  else False
xvfb_str = "" if not use_xvfb else "xvfb-run --auto-servernum --server-num=1 "
eval_freq = 5000 if envtype == "classic_control" else eval_freq
eval_freq = 50000
eval_cb = "" if not no_eval else " --no_eval_callback"

if use_cpu:
    partition = f"cpu_{tnt_cpu_partition}" if not on_luis else "amo"
    gres = None
else:
    # use gpu
    partition = "gpu_normal" if not on_luis else "gpu"
    gres = "gpu:1"
mail_user = "benjamin@tnt.uni-hannover.de" if not on_luis else "benjamins@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
mem_per_cpu = "2000M" if "Racing" not in env else "8000M"
basecommand = f"{xvfb_str}python {runfile} --num_contexts 100 --steps {n_timesteps} " \
              f"--add_context_feature_names_to_logdir --hp_file training/hyperparameters/ppo.yml " \
              f"--use_cgate {use_cgate}"
cpus_per_task = "16"
pre_command = ""
post_command = ""
runcommands_file_precommand = ""
if on_luis:
    luis_template_fn = "utils/LUIS_template.sh"
    conda_env_name = "carl"
    project_name = "CARL"
    slurmout_dir = os.path.join("$BIGWORK", project_name, "slurmout")  # need to define it manually so far :( # TODO define once per sh, maybe in runcommands.sh
    slurmout_dir = os.path.join("/bigwork/", luis_user_name, project_name, "slurmout")  # cannot use environment variables in #SBATCH definitions :(
    output_filename = os.path.join(slurmout_dir, "slurm-%j.out")
    with open(luis_template_fn, 'r') as file:
        content = file.read()
    pre_command = content.format(conda_env_name=conda_env_name, project_name=project_name, branch_name=branch_name, slurmout_dir=slurmout_dir)
    runcommands_file_precommand = "git pull\n"

    # remove everything in runscripts/generated
    runfile_dir = "runscripts/generated"
    files = glob.glob(os.path.join(runfile_dir, "*.sh"))
    for file in files:
        os.remove(file)

outdir = f"results/singlecontextfeature_{default_sample_std_percentage}{hide_context_dir_str}/{envtype}/{env}"
hide_context_dir_str = "contexthidden" if hide_context else "contextvisible"
state_context_features_str = "changing" if state_context_features is not None else ""
outdir = os.path.join(outdirbase, f"{exptype}/{envtype}")
if on_luis:
    outdir = os.path.join("$WORKING_DIR", outdir)
basecommand += f" --outdir {outdir}  --num_workers {cpus_per_task} --build_outdir_from_args"
basecommand += f" --eval_freq {eval_freq} --seed $SLURM_ARRAY_TASK_ID " \
               f"--scale_context_features no  --vec_env_cls {vec_env_cls}  --agent {agent} "
basecommand += eval_cb
if use_xvfb:
    basecommand += " --use_xvfb"
if state_context_features is not None:
    basecommand += f" --state_context_features {state_context_features} "
if "Racing" in env:
    basecommand += f" --context_file {context_file} "
is_hide_context_in_iter = np.any([el["name"] == "hide_context" for el in iteration_list])
if not is_hide_context_in_iter:
    basecommand += f" --hide_context {hide_context}"
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
    }
if gres is not None:
    slurm_config["gres"] = gres
if on_luis:
    if partition == "amo":
        del slurm_config["gres"]


sbuilder = SlurmBuilder(
    slurm_config=slurm_config,
    pre_command=pre_command,
    base_command=basecommand,
    post_command=post_command,
    output_filename=output_filename,
    runcommands_file_precommand=runcommands_file_precommand,
    iteration_list=iteration_list,
)
sbuilder.build_shfiles()

transfer_fn = "runscripts/transfer_generated.sh"
content = "scp -r runscripts/generated/  nhmlbenc@transfer.cluster.uni-hannover.de:/home/nhmlbenc/repos/CARL/carl/runscripts/"
with open(transfer_fn, 'w') as file:
    file.write(content)


# longer output for squeue
# squeue --format="%.18i %.11P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
