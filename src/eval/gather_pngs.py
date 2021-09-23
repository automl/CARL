import os
from pathlib import Path
path = "results"

env_name = "CARLLunarLanderEnv"

results = []
for root, dirs, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith(".png") and "step" in filename:
            full_filename = Path(os.path.join(root, filename))
            exp_def = full_filename.parts[1].split("_")
            relstd = float(exp_def[1])
            hidecontext = False
            if len(exp_def) >= 3:
                hidecontext = True

            result = {
                "filename": full_filename,
                "hidecontext": hidecontext,
                "relstd": relstd
            }
            results.append(result)
