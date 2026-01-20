"""
    tests / path.py
    ---------------
    Test CLI script for path model which includes various models of generative ai
    models to model the UAV trajectory based on the data from the link model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from src.models.chanmod import ChannelModel
from data.loader import DataLoader, shuffle_and_split
from tests.debug.parser import build_parser, mainrunner, CommandSpec



def test_build_path_model(args: argparse.Namespace):
    c = ChannelModel()

# ============================================================
#       Main Script
# ============================================================

SEED = [{"flags": ["--seed","-s"], "kwargs": {"type": int, "default": 42}}]


@mainrunner
def main():
    p = build_parser([
        CommandSpec(name="build", help="", handler=test_build_path_model, args=[])
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__": main()
