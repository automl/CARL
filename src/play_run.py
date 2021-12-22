from src.train import get_parser, main
if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    args.agent = "PPO"
    args.env = "CARLMountainCarEnv"
    args.state_context_features = "changing_context_features"
    args.steps = 300000
    args.seed = 2
    args.no_eval_callback = True
    args.num_envs = 8
    args.use_xvfb = False
    args.hide_context = False
    args.context_feature_args = ["g", "m"]
    # args.context_file = "envs/box2d/parking_garage/context_set_all.json"
    print(args)
    main(args, unknown_args, parser)
