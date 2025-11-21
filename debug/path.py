from __future__ import annotations

import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import tensorflow as tf

from logs.logger import Logger, LogLevel
from data.get_data import get_shuffled_city_data
from src.models.chanmod import ChannelModel



def train_path(args: argparse.ArgumentParser, logger: Logger):
    dtr, dts = get_shuffled_city_data(
        cities=args.cities,validation_ratio=args.val_ratio, logger=logger
    )

    model = ChannelModel(directory=args.cities, seed=args.seed)

    model.path.build()
    # model.path.load()
    model.path.fit(
        dtr=dtr, dts=dts, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    # model.path.save()



def build_parser() -> argparse.ArgumentParser:
    """
    Forming the argument parser
    """
    parser = argparse.ArgumentParser(description="CLI tester loader class")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')

    def add_common(p: argparse.ArgumentParser):
        p.add_argument(
            "--loglevel", type=str, default="INFO", help="Loglevel assignment",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        p.add_argument(
            "--cities", type=str, default="beijing",
            choices=["beijing", "boston", "moscow", "london", "tokyo"]
        )
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--batch-size", type=int, default=512)
        p.add_argument("--learning-rate", type=float, default=1e-4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--val-ratio", type=float, default=0.10)

    for cmd, help_text in  [
        ("train", "Training the Path model with designated generative model")
    ]: add_common(subparsers.add_parser(cmd, help=help_text))
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger("path-cli", to_disk=False, level=LogLevel.INFO)

    commands = {
        "train": train_path,
    }

    try: commands[args.command](args, logger)
    finally: Logger.shutdown_all()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
