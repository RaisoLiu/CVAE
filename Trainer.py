import argparse
import os
import resource

import yaml

from modules import VAE_Model

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


def parse_milestones(args):
    if not args.milestones:
        args.milestones = [args.num_epoch // 2, args.num_epoch - 1]
        print(f"Milestones were empty, set to: {args.milestones}")
    else:
        args.milestones = [int(x) for x in args.milestones.split(",")]
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="initial learning rate"
    )
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--store_visualization",
        action="store_true",
        help="If you want to see the result while training",
    )
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument(
        "--save_root", type=str, required=True, help="The path to save your data"
    )
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument(
        "--num_epoch", type=int, default=100, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=1, help="Save checkpoint every seted epoch"
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Part of the training dataset to be trained",
    )
    parser.add_argument(
        "--train_vi_len", type=int, default=128, help="Training video length"
    )
    parser.add_argument(
        "--val_vi_len", type=int, default=630, help="valdation video length"
    )
    parser.add_argument(
        "--frame_H", type=int, default=32, help="Height input image to be resize"
    )
    parser.add_argument(
        "--frame_W", type=int, default=64, help="Width input image to be resize"
    )

    # Data augmentation parameters
    parser.add_argument(
        "--use_random_crop",
        action="store_true",
        help="Whether to use random crop data augmentation",
    )

    # Module parameters setting
    parser.add_argument(
        "--F_dim", type=int, default=128, help="Dimension of feature human frame"
    )
    parser.add_argument(
        "--L_dim", type=int, default=32, help="Dimension of feature label frame"
    )
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of the Noise")
    parser.add_argument(
        "--D_out_dim",
        type=int,
        default=192,
        help="Dimension of the output in Decoder_Fusion",
    )

    # Teacher Forcing strategy
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="The initial teacher forcing ratio"
    )
    parser.add_argument(
        "--tfr_sde",
        type=int,
        default=25,
        help="The epoch that teacher forcing ratio start to decay",
    )
    parser.add_argument(
        "--tfr_d_step",
        type=float,
        default=0.1,
        help="Decay step that teacher forcing ratio adopted",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="The path of your checkpoints"
    )

    # Training Strategy
    parser.add_argument("--fast_train", action="store_true")
    parser.add_argument(
        "--fast_partial",
        type=float,
        default=0.4,
        help="Use part of the training data to fasten the convergence",
    )
    parser.add_argument(
        "--fast_train_epoch",
        type=int,
        default=5,
        help="Number of epoch to use fast train mode",
    )

    # Kl annealing stratedy arguments
    parser.add_argument("--kl_anneal_type", type=str, default="Cyclical", help="")
    parser.add_argument("--kl_anneal_cycle", type=int, default=10, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    # Learning rate scheduler milestones
    parser.add_argument(
        "--milestones",
        type=str,
        default="2,4",
        help="Comma-separated list of epochs for learning rate decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor of learning rate decay",
    )

    # Mixed precision training
    parser.add_argument(
        "--no-use_amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision training",
    )

    args = parser.parse_args()
    args = parse_milestones(args)

    main(args)
