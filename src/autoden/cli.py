# Why does this file exist, and why not put this in `__main__`?
#
# You might be tempted to import things from `__main__` later,
# but that will cause problems: the code will get executed twice:
#
# - When you run `python -m autoden` python will execute
#   `__main__.py` as a script. That means there won't be any
#   `autoden.__main__` in `sys.modules`.
# - When you import `__main__` it will get executed again (as a module) because
#   there's no `autoden.__main__` in `sys.modules`.

"""Module that contains the command line application."""

from pathlib import Path

import argparse

import imageio.v3 as iio

import numpy as np

from autoden import __version__
from autoden.models.config import NetworkParamsUNet
from autoden import DIP

DEFAULT_TV_VAL = 1e-6
DEFAULT_EPOCHS = 2_000


def get_parser() -> argparse.ArgumentParser:
    """
    Return the CLI argument parser.

    Returns:
        An argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog="autoden",
        description="Denoise the given images, using deep-learning-based unsupervised or self-supervised algorithms.",
    )
    parser.add_argument("algorithm", choices=["N2N", "N2V", "DIP"], help="Denoising algorithm to use")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help=f"Number of epochs to use, by default {DEFAULT_EPOCHS}.",
        metavar="E",
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--unet-levels",
        "-l",
        type=int,
        help=f"Number of UNet levels to use, by default: {NetworkParamsUNet.DEFAULT_LEVELS}.",
        default=NetworkParamsUNet.DEFAULT_LEVELS,
        metavar="L",
    )
    parser.add_argument(
        "--unet-features",
        "-f",
        type=int,
        help=f"Number of UNet features to use, by default: {NetworkParamsUNet.DEFAULT_FEATURES}.",
        default=NetworkParamsUNet.DEFAULT_FEATURES,
        metavar="F",
    )
    parser.add_argument(
        "--regularization",
        "-r",
        type=float,
        help=f"Total Variation regularization value, by default: {DEFAULT_TV_VAL}.",
        default=DEFAULT_TV_VAL,
        metavar="R",
    )
    parser.add_argument("src_file", nargs="+", help="Path of each input image.", type=argparse.FileType("rb"))
    parser.add_argument("dst_file", help="Path of the output image.", type=argparse.FileType("wb"))
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(args: list[str] | None = None) -> int:
    """
    Run the main program.

    This function is executed when you type `autoden` or `python -m autoden`.

    Arguments:
        args: Arguments passed from the command line.

    Returns:
        An exit code.
    """
    parser = get_parser()
    opts = parser.parse_args(args=args)
    # print(opts)  # noqa: WPS421 (side-effect in main is fine)

    inp_imgs = [iio.imread(f) for f in opts.src_file]
    if any(x.ndim > 2 for x in inp_imgs):
        print("Color images not supported, yet.")
        return 1

    net_pars = NetworkParamsUNet(n_levels=opts.unet_levels, n_features=opts.unet_features)
    if opts.algorithm.upper() == "DIP":
        algo = DIP("", network_type=net_pars, save_epochs=False, reg_tv_val=opts.regularization)
        inp_img = algo.train_unsupervised(np.stack(inp_imgs, axis=0), epochs=opts.epochs)
        out_img = algo.infer(inp_img)
        iio.imwrite(opts.dst_file, out_img)
    else:
        print(f"Not implemented support for algorithm {opts.algorithm} in command-line, yet.")
        return 1
    return 0
