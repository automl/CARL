import json
from pathlib import Path
import configargparse
import sys
sys.path.append("../../..")
from carl.experiments.hpo.smac.smac_plot_trajectory import gather_smac_data, extract_incumbents
from carl.experiments.hpo.smac.hp_opt import carl_from_cfg, get_parser


def get_retrain_parser() -> configargparse.ArgumentParser:
    """
    Creates new argument parser for running baselines.

    Returns
    -------
    parser : argparse.ArgumentParser

    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.ConfigparserConfigFileParser
    )
    parser.add_argument(
        "--expdir",
        type=str,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1e6
    )
    parser.add_argument(
        "--incumbent_idx",
        type=int,
        default=0
    )
    return parser


if __name__ == '__main__':
    # outdir = "/home/benjamin/Dokumente/code/tmp/carl/src/results/optimized/classic_control/CARLCartPoleEnv/0.1_contexthidden"
    aux_parser = get_retrain_parser()
    aux_args, unknown_args = aux_parser.parse_known_args()
    expdir = aux_args.expdir
    steps = aux_args.steps
    incumbent_idx = aux_args.incumbent_idx

    convert_sec_to_hours = True
    key_time = "wallclock_time"
    key_performance = "cost"
    key_group = "exp_source"

    data = gather_smac_data(outdir=expdir, key_group=key_group)
    incumbents = extract_incumbents(data=data, key_group=key_group, key_performance=key_performance)

    incumbent = incumbents.iloc[incumbent_idx]
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
    print("final_ep_mean_reward", -ret)
