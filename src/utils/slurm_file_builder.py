from slurmbuilder.slurmbuilder import SlurmBuilder
import src.envs as envs

######################################
job_name = "CARL"
env = "CARLPendulumEnv"
envtype = "classic_control"
default_sample_std_percentage = 0.75
hide_context = True
vec_env_cls = "DummyVecEnv"
agent = "PPO"
n_timesteps = 1000000
state_context_features = None  # "changing_context_features"
no_eval = True
hp_opt = True
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

partition = "gpu_normal"
mail_user = "benjamin@tnt.uni-hannover.de"
output_filename = "slurmout/slurm-%j.out"
time = "24:00:00"
mem_per_cpu = "2000M" if "racing" not in env else "8000M"
basecommand = f"{xvfb_str}python {runfile} --num_contexts 100 --steps {n_timesteps} " \
              f"--add_context_feature_names_to_logdir --hp_file training/hyperparameters/hyperparameters_ppo.yml"
cpus_per_task = "1"

env_defaults = getattr(envs, f"{env}_defaults")
outdir = f"results/singlecontextfeature_{default_sample_std_percentage}{hide_context_dir_str}/{envtype}/{env}"
hide_context_dir_str = "contexthidden" if hide_context else "contextvisible"
state_context_features_str = "changing" if state_context_features is not None else ""
outdir = f"results/{exptype}/{envtype}/{env}/{default_sample_std_percentage}_{state_context_features_str}{hide_context_dir_str}"
basecommand += f" --outdir {outdir}  --num_workers {cpus_per_task} --default_sample_std_percentage {default_sample_std_percentage}"
basecommand += f" {hide_context_cmd_str} --eval_freq {eval_freq} --seed $SLURM_ARRAY_TASK_ID " \
               f"--scale_context_features no  --vec_env_cls {vec_env_cls}  --agent {agent} "
basecommand += eval_cb
if state_context_features is not None:
    basecommand += f"--state_context_features {state_context_features}"
context_features = list(env_defaults.keys())
seeds = [0, 1, 2, 3, 4]
array = f"{min(seeds)}-{max(seeds)}"

sbuilder = SlurmBuilder(
    job_name=job_name,
    mail_user=mail_user,
    base_command=basecommand,
    time=time,
    partition=partition,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    array=array,
    output_filename=output_filename,
    iteration_list=[
        {
            "name": "env",
            "id": "env",
            "values": [env]
        },
        {
            "name": "context_feature_args",
            "id": "cfargs",
            "values": ['g']  # list(env_defaults.keys())  # None trains only with the default context
        }
    ]
)
sbuilder.build_shfiles()
