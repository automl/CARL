import os
import glob
import numpy as np
import src.envs as envs
from pathlib import Path
import itertools


job_name = "CARL"
env = "CARLCartPoleEnv"
agent = "DQN"
n_timesteps = 500_000
state_context_features = "changing_context_features"
follow_evaluation_protocol = True
outdir = "results"

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
    },
    {
        "name": "seed",
        "id": "s",
        "values": [0, 1, 2, 3, 4]
    }
]

runfile = "train.py"
eval_freq = 50000
basecommand = f"python {runfile} --num_contexts 100 --steps {n_timesteps} " \
              f"--add_context_feature_names_to_logdir --scale_context_features no "
basecommand += f" --outdir {outdir} --build_outdir_from_args"
basecommand += f" --eval_freq {eval_freq} --agent {agent} "
if state_context_features is not None:
    basecommand += f" --state_context_features {state_context_features} "


commands = []

arg_names = [v["name"] for v in iteration_list]
arg_ids = [v["id"] for v in iteration_list]
arg_vals = [v["values"] for v in iteration_list]

all_combos = itertools.product(*arg_vals)
for combo in all_combos:
    for arg_id, value in zip(arg_ids, combo):
        if isinstance(value, (list, tuple, np.ndarray)):
            v = value[0]
        else:
            v = value
    kwargs = {arg_name: value for arg_name, value in zip(arg_names, combo)}
    maincommand = basecommand
    for key, val in kwargs.items():
        maincommand += f" --{key} {val}"
    commands.append(maincommand)


fullcommand = "\n".join(commands)

shfilename = Path("runscripts/run_train_evaluation_protocol.sh")
shfilename.parent.mkdir(parents=True, exist_ok=True)
with open(shfilename, 'w') as file:
    file.write(fullcommand)
print(f"Built '{shfilename}'")
print(fullcommand)
