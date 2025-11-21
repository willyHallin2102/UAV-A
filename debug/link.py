"""
    debug/link.py
    -------------
    Extends a cli script for debugging and testing the link model, this model is 
    being reused for any other model, city or generative model.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from data.get_data import get_shuffled_city_data
from src.models.chanmod import ChannelModel

from logs.logger import Logger, LogLevel




def train_link(args: argparse.ArgumentParser, logger: Logger):
    # print("training link model")
    dtr, dts = get_shuffled_city_data(
        cities=args.cities, validation_ratio=args.val_ratio
    )

    model = ChannelModel(directory=args.cities, seed=args.seed)
    model.link.build()
    model.link.fit(
        dtr=dtr, dts=dts, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    model.link.save()




# ---------------========== Building Parser ==========--------------- #

def build_parser() -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="CLI tester for link state predictor")
    subparser = parser.add_subparsers(dest="command", required=True, help="Command to run")

    def add_common(p: argparse.ArgumentParser):
        p.add_argument(
            "--loglevel", type=str, default="INFO", help="loglevel assignment",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )

        p.add_argument(
            "--cities", type=str, default="beijing", help="which city to train on",
            choices=["beijing", "boston", "moscow", "london", "tokyo"]
        )

        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--batch-size", type=int, default=512)
        p.add_argument("--learning-rate", type=float, default=1e-4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--val-ratio", type=float, default=0.10)
    
    for cmd, help_text in [
        ("train", "Training the link state predictor"),
        ("test_model_arch", "Testing the various architecture"),
        ("test_pred", "Test prediction capabilities of the model"),
        ("test_data", "testing if generating data is working")
    ]: add_common(subparser.add_parser(cmd, help=help_text))
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger(name="link-cli", to_disk=False, level=LogLevel.INFO)
    commands = {
        "train": train_link,
    }

    try: commands[args.command](args, logger)
    finally: Logger.shutdown_all()



if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
