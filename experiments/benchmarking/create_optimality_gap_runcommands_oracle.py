import sys
from pathlib import Path
from experiments.carlbench.context_logging import load_wandb_table_to_df
from experiments.common.utils.json_utils import lazy_json_dump
from omegaconf import OmegaConf
from rich import print as printr


if __name__ == "__main__":
    env_name = sys.argv[1]
    
    bash_fn = f"/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/optimality_gap/{env_name}/run_train_oracles.sh"

    outdir = f"./runs/optimality_gap/{env_name}/train_general"
    outdir = Path(outdir)

    target_path = f"/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/optimality_gap/{env_name}/contexts"
    target_path = Path(target_path)

    command_dir = Path(f"/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/optimality_gap/{env_name}/runcommands")

    contexts_paths = list(outdir.rglob("latest-run/files/media/table/train/contexts*json"))
    contexts_paths.sort()
    contexts_list = [load_wandb_table_to_df(p) for p in contexts_paths]

    config_paths = list(outdir.glob("*/.hydra/config.yaml"))
    config_paths.sort()
    config_list = [OmegaConf.load(p) for p in config_paths]
    seed_list = [cfg.seed for cfg in config_list]

    command_fns = []

    for i, contexts in enumerate(contexts_list):
        seed = seed_list[i]
        new_contexts_list = [{j: dict(context)} for j, context in contexts.iterrows()]
        new_contexts_paths = []
        for j, C in enumerate(new_contexts_list):
            # Create contexts sets with single context and save
            context_fn = target_path / f"context_{seed}_{j}.json"
            new_contexts_paths.append(str(context_fn))
            lazy_json_dump(C, context_fn)
            
        # Build runcommand

        # train
        new_contexts_paths_str = ",".join(new_contexts_paths)
        snap_dir = f"./runs/optimality_gap/{env_name}/train_oracle/seed_{seed}"
        command = f"'seed={seed}' 'contexts_train_path={new_contexts_paths_str}' --snap_dir {snap_dir}"
        fullcommand = "python experiments/benchmarking/run_training.py '+experiments=optimality_gap' " + command + " -m" 

        # Save runcommand
        command_fn = command_dir / f"run_{seed}.sh"
        command_fn = Path(command_fn)
        command_fn.parent.mkdir(exist_ok=True, parents=True)
        with open(command_fn, 'w') as file:
            file.write(fullcommand)
        printr(command_fn)
        command_fns.append(command_fn)
        # printr(fullcommand[:260])
    
    main_run_fn = command_dir / "run_train_oracles.sh"
    with open(main_run_fn, 'w') as file:
        file.writelines([f"bash {c}\n" for c in command_fns])
    printr(main_run_fn)