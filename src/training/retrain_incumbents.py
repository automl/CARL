import json
from pathlib import Path
from src.training.smac_plot_trajectory import gather_smac_data, extract_incumbents
from src.training.hp_opt import carl_from_cfg, get_parser

if __name__ == '__main__':
    outdir = "/home/benjamin/Dokumente/code/tmp/carl/src/results/optimized/classic_control/CARLCartPoleEnv/0.1_contexthidden"
    steps = 1000

    convert_sec_to_hours = True
    key_time = "wallclock_time"
    key_performance = "cost"
    key_group = "exp_source"

    data = gather_smac_data(outdir=outdir, key_group=key_group)
    incumbents = extract_incumbents(data=data, key_group=key_group, key_performance=key_performance)

    for i in range(len(incumbents)):
        incumbent = incumbents.iloc[i]
        trial_setup_fn = incumbent['exp_source']
        with open(trial_setup_fn, 'r') as file:
            trial_setup = json.load(file)

        parser = get_parser()
        args, unknown_args = parser.parse_known_args()
        ignore = ['steps']
        for k, v in trial_setup.items():
            if k not in ignore:
                setattr(args, k, v)
        args.build_outdir_from_args = False
        args.dont_add_agentseed_to_logdir = True
        args.add_context_feature_names_to_logdir = False
        outdir = Path(trial_setup_fn).parent / 'train_inc'
        args.outdir = str(outdir)
        cfg = incumbent["incumbent"]
        args.steps = steps
        ret = carl_from_cfg(cfg=cfg, seed=args.seed, budget=args.steps, parser=parser, args=args)
        break
