from src.train import get_parser, main
if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    args.agent = "PPO"
    args.env = "CARLCartPoleEnv"
    args.state_context_features = "changing_context_features"
    args.steps = 200000
    args.seed = None
    args.no_eval_callback = True
    args.num_envs = 8
    # args.use_xvfb = True
    args.context_feature_args = ["None"]
    print(args)
    main(args, unknown_args, parser)
