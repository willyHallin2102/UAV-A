"""
    debug/loader.py
    ---------------
    Extended CLI tester for data loading and processing functionality
"""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd

from data.data_processing import DataProcessor
from data.file_handlers import HandlerFactory, CsvHandler
from data.loader import DataLoader, shuffle_and_split
from data.get_data import get_city_data, get_shuffled_city_data

from logs.logger import Logger, LogLevel, get_loglevel



def test_load(args: argparse.Namespace, logger: Logger):
    logger.info(f"Loading data for city {args.city}")
    # with logger.time_block(f"Loading {args.city} data", level=LogLevel.INFo):
    with logger.time_block(f"Loading {args.city} data", level=get_loglevel(args.loglevel)):
        data = get_city_data(cities=args.city, logger=logger)
    
    logger.info(
        f"Loaded {len(data)} columns with shapes: "
        f"{', '.join(f'{key}: {value.shape}' for key, value in data.items())}"
    )

    if args.sample:
        logger.info("sample data")
        for key, array in data.items():
            logger.info(f"{key}: {array[:2] if len(array) > 0 else 'empty'}")



# ---------------========== Testing all scripts ==========--------------- #

def run_all_tests(args: argparse.Namespace, logger: Logger):
    """Run all tests in sequence"""
    tests = [
        test_load
    ]
    
    for f in tests:
        logger.info(f"Running test: {f.__name__}")
        try:
            test_func(args, logger)
            logger.info(f"✓ {f.__name__} passed")
        except Exception as e: logger.error(f"✗ {f.__name__} failed: {e}")
        time.sleep(0.1)  # Small delay between tests, useful makes it easier



# ---------------========== Build Argument Parser ==========--------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extended CLI tester for data loading and processing")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')

    def add_common_args(p: argparse.ArgumentParser):
        p.add_argument("--loglevel", type=str, default="DEBUG", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
        p.add_argument("--city", type=str, default="beijing",
                       choices=["beijing", "boston", "london", "moscow", "tokyo"],
                       help="City to test with")
        p.add_argument(
            "--to-disk", action="store_true",
            help="Enable disk logging (default: console only)"
        )
        p.add_argument(
            "--no-json", action="store_true",
            help="Disable JSON formatting (use console formatting)"
        )
    
    load_parser = subparsers.add_parser("load", help='Test basic data loading')
    add_common_args(load_parser)
    load_parser.add_argument("--sample", action="store_true", help="Show sample data")
    

    return parser



# ---------------========== Main Script ==========--------------- #

def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger(
        name="logger-cli", level=get_loglevel(args.loglevel),
        json_format=not args.no_json, use_console=True,
        to_disk=args.to_disk
    )

    commands = {
        "load"      : test_load,
    }

    try:
        logger.info(f"Starting test: {args.command}")
        commands[args.command](args, logger)
        logger.info(f"Test completed: {args.command}")
    except Exception as e:
        logger.critical(f"Test failed: {e}")
        raise
    finally:
        Logger.shutdown_all()


# ---------------========== Running script ==========--------------- #

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
    except Exception as e:
        print(f"CLI failed: {e}")
        Logger.shutdown_all()
        sys.exit(1)
