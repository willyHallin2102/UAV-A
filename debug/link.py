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
from typing import Any, Tuple

from data.get_data import get_shuffled_city_data
from src.config.data import LinkState, DataConfig
from src.models.chanmod import ChannelModel

from logs.logger import Logger, LogLevel

#
#
#
#
#

def _create_test_data(n_samples: int=1000, seed: int=42):
    np.random.seed(seed)
    dvec = np.random.uniform(-1000, 1000, (n_samples, 3))

    rx_type = np.random.choice(["0", "1"], n_samples)

    # Generate link states (0 for NLoS, 1 for LoS)
    # Simple heuristic: higher altitude and closer horizontal distance = more likely LoS
    horizontal_distance = np.linalg.norm(dvec[:, :2], axis=1)
    altitude = dvec[:, 2]

    # Simple rule: if altitude > 50 and horizontal_dist < 200, likely LoS
    link_state = np.where(
        (altitude>50) & (horizontal_distance<200),
        LinkState.LOS, LinkState.NLOS
    )
    return {
        'dvec': dvec.astype(np.float32),
        'rx_type': rx_type,
        'link_state': link_state
    }

#
#
#
#
#

def test_model_construction(args: argparse.ArgumentParser, logger: Logger):
    architectures = [(64, 32), (128, 64, 32), (256, 128, 64, 32)]
    logger.info("Initialize the Link State Predictor architectures")
    for architecture in architectures:
        print(f"\nTesting architecture: {architecture}")
        data_cfg = DataConfig(rx_types=["0", "1"], n_unit_links=architecture)
        model = ChannelModel(directory=args.cities, config=data_cfg, seed=args.seed)
        model.link.build()



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



def test_predictions(args: argparse.ArgumentParser, logger: Logger):
    """
    """
    logger.info("Initialize Prediction test")
    model = ChannelModel(directory=args.cities)
    model.link.load()

    # dtr = create_test_data(1000)
    # dts = create_test_data(200)
    test_dvec = np.array([[50, 50, 100], [10, 10, 20], [200, 200, 10]], dtype=np.float32)
    test_rx_type = np.array([0, 1, 0])

    probabilities = model.link.predict(test_dvec, test_rx_type, batch_size=32)
    sampled_states = model.link.sample_link_state(test_dvec, test_rx_type)





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

    logger = Logger("link-cli", to_disk=False, level=LogLevel.INFO)
    

    commands = {
        "train": train_link,
        "test_model_arch": test_model_construction,
        "test_pred": test_predictions,
        "test_data": _create_test_data
    }

    try: commands[args.command](args, logger)
    finally: Logger.shutdown_all()



if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
