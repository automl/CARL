from src.train import get_parser, main
if __name__ == '__main__':
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    args.agent = "DDPG"
    args.env = "CARLPendulumEnv"
    args.state_context_features = "changing_context_features"
    args.steps = 300000
    args.seed = 2
    args.no_eval_callback = True
    args.hide_context = False
    print(args)
    main(args, unknown_args, parser)
