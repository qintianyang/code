import argparse


def get_config():
    # Constants for allowed values
    ALLOWED_EVALUATIONS = [
        "eeg",
        "correct_watermark",
        # "wrong_watermark",
        # "new_watermark",
    ]
    ALLOWED_EXPERIMENTS = [
       "from_scratch_ep3"
       "from_scratch_ep3_EWE",
       "from_scratch_RS",
       "from_scratch_EEF",
       "from_scratch_DI"

    ]
    TRAINING_MODES = ["skip", "quick", "full"]
    ARCHITECTURES = ["CCNN", "EEGNet", "TSCeption"]
    VERBOSE_LEVELS = ["info", "debug", "warning", "error", "critical"]

    parser = argparse.ArgumentParser(
        description="Configure and run experiments for watermarking EEG-based neural networks.",
    )

    # Experiment Configuration
    config_group = parser.add_argument_group("Experiment Configuration")
    config_group.add_argument(
        "--experiment",
        # required=True,
        default= "from_scratch_EEF" ,
        # choices=ALLOWED_EXPERIMENTS,
        help="Choose one experiment from the above experiments.",
    )
    config_group.add_argument(
        "--evaluate",
        nargs="+",
        default=ALLOWED_EVALUATIONS, 
        # choices=ALLOWED_EVALUATIONS,
        metavar="DIMENSION",
        help=f"Choose any number of dimensions to evaluate from {{{','.join(ALLOWED_EVALUATIONS)}}}.",
    )
    config_group.add_argument(
        "--architecture",
        # required=True,
        default="CCNN",
        choices=ARCHITECTURES,
    )

    # Training Parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--training_mode",
        default="full",
        choices=TRAINING_MODES,
        help="Skip training, quick training of only 1 fold, or full training of all folds.",
    )
    train_group.add_argument("--batch", type=int,default=64, help="Batch size for training.")
    train_group.add_argument("--epochs", type=int,default=200, help="Number of training epochs.")
    train_group.add_argument("--lrate", type=float, default=0.0005,help="Learning rate for training.")
    train_group.add_argument(
        "--update_lr_by",
        type=float,
        default=1.0,
        metavar="x",
        help="Multiply learning rate by x every n epochs. Default x: 1.0",
    )
    train_group.add_argument(
        "--update_lr_every",
        type=int,
        default=10,
        metavar="n",
        help="Multiply learning rate by x every n epochs. Default n: 10",
    )
    train_group.add_argument(
        "--update_lr_until",
        type=float,
        default=1e-5,
        metavar="ε",
        help="Update learning until it's out of [ε, 1.0]. Default ε: 1e-5",
    )
    train_group.add_argument(
        "--folds",
        type=int,
        default=10,
        metavar="k",
        help="Number of k-fold cross-validation splits. Default k: 10.",
    )

    # Path Configuration
    path_group = parser.add_argument_group("Path Configuration")
    path_group.add_argument(
        "--data_path",
        metavar="PATH",
        default="/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python",
        help="Path to processed data directory. Default: './data/data_preprocessed_python'.",
    )
    path_group.add_argument(
        "--base_models_dir",
        metavar="DIR",
        default="model",
        help="Directory containing base models for experiments.",
    )

    # Experiment-Specific Parameters
    exp_params_group = parser.add_argument_group("Experiment-Specific Parameters")
    exp_params_group.add_argument(
        "--pruning_method",
        default="random",
        choices=["random", "ascending", "descending"],
        help="Random, ascending (nullify least-valued nodes), descending (nullify most-valued nodes).",
    )
    exp_params_group.add_argument(
        "--pruning_mode",
        default="linear",
        choices=["linear", "exponential"],
        help="Linear increments pruning percentage by δ till it reaches 100, whereas exponential multiplies by it.",
    )
    exp_params_group.add_argument(
        "--pruning_delta",
        metavar="δ",
        type=float,
        default=5,
        # 对于线性的增长 推荐的大小是5 对于是指数增长 推荐的大小是
        help="Increment/multiply pruning percent by δ. Recommended: 5 for linear, 1.25 for exponential.",
    )
    exp_params_group.add_argument(
        "--fine_tuning_mode",
        choices=["ftll", "ftal", "rtll", "rtal"],
        default="rtal",
        help="FTLL (fine-tune last layer), FTAL (fine-tune all layers), RTLL (retrain last layer), and RTAL (retrain all layers).",
    )
    exp_params_group.add_argument(
        "--transfer_learning_mode",
        choices=["added", "dense", "all"],
        default="dense",
        help="Add two dense layers then perform fine tuning. Added (fine-tune added layers), dense (fine-tune dense layers), and all (fine-tune all layers).",
    )

    # Other Parameters
    other_group = parser.add_argument_group("Other Parameters")
    other_group.add_argument(
        "--seed",
        type=int,
        default= 40,
        help="Seed for reproducibility.",
    )
    other_group.add_argument(
        "--verbose",
        choices=VERBOSE_LEVELS,
        default="debug",
        help="How much information to log. Default is 'info'.",
    )

    # Parse and validate arguments
    args = vars(parser.parse_args())
    validate_arguments(parser, args)

    return args


def validate_arguments(parser, args):
    EXPERIMENTS_REQUIRING_BASE_MODELS = [
        "pruning",
        "pretrain",
        "fine_tuning",
        "quantization",
        "new_watermark",
        "transfer_learning",
        "feature_attribution",
    ]

    if args["experiment"] in ["quantization", "pruning", "feature_attribution"]:
        args["training_mode"] = "skip"
    if args["experiment"].startswith("show_stats"):
        args["training_mode"] = "skip"

    if args["training_mode"] != "skip":
        require_args(
            parser,
            args,
            ["batch", "epochs", "lrate"],
            "are required when training_mode is not 'skip'.",
        )

    if args["experiment"] in EXPERIMENTS_REQUIRING_BASE_MODELS:
        require_arg(
            parser, args, "base_models_dir", "is required for this experiment type."
        )

    validation_rules = {
        "pruning": ["pruning_method", "pruning_mode", "pruning_delta"],
        "transfer_learning": ["transfer_learning_mode"],
        "fine_tuning": ["fine_tuning_mode"],
    }

    if args["experiment"] in validation_rules:
        required_args = validation_rules[args["experiment"]]
        require_args(
            parser,
            args,
            required_args,
            f"are required for {args['experiment']} experiments.",
        )


def require_args(parser, args, arg_names, message):
    missing = [f"--{arg}" for arg in arg_names if not args.get(arg)]
    if missing:
        parser.error(f"{', '.join(missing)} {message}")


def require_arg(parser, args, arg_name, message):
    if not args.get(arg_name):
        parser.error(f"--{arg_name} {message}")
