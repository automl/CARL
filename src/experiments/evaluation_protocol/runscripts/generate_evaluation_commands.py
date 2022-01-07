import glob
import os
from pathlib import Path
import itertools

on_slurm = False
basecommand = "python evaluate_on_protocol.py "

# Model Filenames
path = "results/CARLCartPoleEnv"
model_fnames = glob.glob(os.path.join(path, "**", "model.zip"), recursive=True)

# Evaluation Protocols
evaluation_protocols = ['train', 'test_interpolation', 'test_interpolation_combinatorial', 'test_extrapolation_single', 'test_extrapolation_all']

if on_slurm:
    # For slurm clusters (config configs/slurm.yaml accordingly)
    evaluation_protocols_string = ",".join(evaluation_protocols)
    model_fnames_string = ",".join(model_fnames)
    fullcommand = f"{basecommand} --multirun context_distribution_type={evaluation_protocols_string} model_fname={model_fnames_string}"
else:
    # Sequential
    arg_vals = [evaluation_protocols, model_fnames]
    all_combos = itertools.product(*arg_vals)
    commands = []
    names = ["context_distribution_type", "model_fname"]
    for combo in all_combos:
        maincommand = basecommand
        for name, value in zip(names, combo):
            maincommand += f" {name}={value} "
        commands.append(maincommand)
    fullcommand = "\n".join(commands)

shfilename = Path("run_eval_evaluation_protocol.sh")
shfilename.parent.mkdir(parents=True, exist_ok=True)
with open(shfilename, 'w') as file:
    file.write(fullcommand)
print(f"Built '{shfilename}'")
print(fullcommand)