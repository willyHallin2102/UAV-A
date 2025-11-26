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



def test_preprocessing(args: argparse.Namespace, logger: Logger):
    logger.info("Testing DataProcessor functionality directly")

    # Create Sample-data: create data that matches the schema
    sample_data = {
        'dvec'      : [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        'rx_type'   : [1, 2, 1],
        'link_state': [0, 1, 0],
        'los_pl'    : [10.5, 11.2, 9.8],
        'los_ang'   : [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        'los_dly'   : [1.1, 1.2, 1.3],
        'nlos_pl'   : [[20.1, 20.2], [21.1, 21.2], [22.1, 22.2]],
        'nlos_ang'  : [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
        'nlos_dly'  : [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]]
    }

    df = pd.DataFrame(sample_data)
    processor = DataProcessor(logger)

    with logger.time_block("DataProcessor chunk processing", level=get_loglevel(args.loglevel)):
        processed = processor.process_chunk(df)
    
    logger.info("Processed data-types:")
    for key, array in processed.items():
        logger.info(f"\t{key}: {array.dtype} -- shape: {array.shape} -- sample: {array[:1]}")


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



def test_load_shuffled(args: argparse.Namespace, logger: Logger):
    logger.info(f"Loading shuffled data for city: {args.city}")
    with logger.time_block(f"Loading shuffled {args.city} data", level=get_loglevel(args.loglevel)):
        train_data, val_data = get_shuffled_city_data(args.city, logger=logger, validation_ratio=args.val_ratio)

    logger.info(
        f"Train data: {len(train_data)} columns, shapes: "
        f"{', '.join(f'{k}: {v.shape}' for k, v in train_data.items())}"
    )
    logger.info(
        f"Validation data: {len(val_data)} columns, shapes: "
        f"{', '.join(f'{k}: {v.shape}' for k, v in val_data.items())}"
    )



def test_load_summary(args: argparse.Namespace, logger: Logger):
    logger.info(f"Loading data summary for city `{args.city}`")

    with logger.time_block(f"Loading {args.city} data", level=get_loglevel(args.loglevel)):
        data = get_city_data(cities=args.city, logger=logger)
    
    logger.info(f"--- SUMMARY REPORT FOR CITY `{args.city}` ---")

    total_cols = len(data)
    total_rows = len(next(iter(data.values())))
    logger.info(f"Rows: {total_rows:,} | Columns: {total_cols}")

    # per-column summary
    for key, array in data.items():
        if isinstance(array, list): array = np.asarray(array, dtype=object)

        logger.info(f"\nColumn: {key}")
        logger.info(f"\tshape: {array.shape}")
        logger.info(f"\tdtype: {array.dtype}")

        if array.dtype == object:
            empty_count = sum((
                x is None or (hasattr(x, '__len__') and len(x) == 0)
            ) for x in array)
            logger.info(f"\tempty entries: {empty_count} ({100 * empty_count / len(array):.3f}%)")
        else:
            # NaN --count
            nan_count = np.isnan(array).sum() if np.issubdtype(array.dtype, np.floating) else 0
            if nan_count: logger.info(f"NaN entries: `{nan_count}`")
        
        # Provide a small safe sample
        sample_size = min(2, len(array))
        sample = array[:sample_size]
        logger.info(f"  sample: {sample}")

        # For numeric columns
        if np.issubdtype(array.dtype, np.number):
            logger.info(f"  min: {array.min()}  max: {array.max()}")



def test_load_profile(args: argparse.Namespace, logger: Logger):
    logger.info(f"Profiling loaded data for city: {args.city}")

    with logger.time_block(f"Loading {args.city} data", level=get_loglevel(args.loglevel)):
        data = get_city_data(args.city, logger=logger)

    logger.info(f"--- PROFILE REPORT FOR `{args.city}` ---")

    # Global stats
    total_rows = len(next(iter(data.values())))
    logger.info(f"Total rows: {total_rows:,}")

    for key, array in data.items():
        if isinstance(array, list): array = np.asarray(array, dtype=object)

        logger.info(f"\nProfiling column: {key}")
        logger.info(f"  dtype: {array.dtype}, shape: {array.shape}")

        # Memory footprint
        try: mem = array.nbytes
        except Exception: mem = "object / variable-sized"
        logger.info(f"  memory: {mem}")

        # Object columns (nested arrays)
        if array.dtype == object:
            # Compute distribution of element lengths
            lengths = []
            for x in array[:5000]:  # keep it fast
                try: lengths.append(len(x))
                except Exception: lengths.append(0)

            unique_lengths = sorted(set(lengths))
            logger.info(f"\telement length distribution (first 5k rows): {unique_lengths[:10]}")

        # Numeric profiling
        if np.issubdtype(array.dtype, np.number):
            logger.info(f"\tmin/max: {array.min()} / {array.max()}")

            # Basic stats without overwhelming logs
            logger.info(f"\tmean: {array.mean():.5f}, std: {array.std():.5f}")

        # Show random sample row
        idx = np.random.randint(0, len(array))
        logger.info(f"\trandom sample at {idx}: {array[idx]}")


# ---------------========== Testing all scripts ==========--------------- #

def run_all_tests(args: argparse.Namespace, logger: Logger):
    """Run all tests in sequence"""
    tests = [
        test_preprocessing, 
        test_load, test_load_shuffled,
        test_load_summary, test_load_profile
    ]
    
    for f in tests:
        logger.info(f"Running test: {f.__name__}")
        try:
            test_func(args, logger)
            logger.info(f"{f.__name__} passed")
        except Exception as e: logger.error(f"{f.__name__} failed: {e}")
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
    
    processor_parser = subparsers.add_parser("processor", help="testing the data-processor directly")
    add_common_args(processor_parser)

    load_parser = subparsers.add_parser("load", help='Test basic data loading')
    add_common_args(load_parser)
    load_parser.add_argument("--sample", action="store_true", help="Show sample data")
    
    shuffle_parser = subparsers.add_parser("shuffle-load", help="Test load shuffled data")
    add_common_args(shuffle_parser)
    shuffle_parser.add_argument("--val-ratio", action="store_true", default=0.2, help="validation portion")

    summary_parser = subparsers.add_parser("summary", help="Load and provide summary")
    add_common_args(summary_parser)

    profile_parser = subparsers.add_parser("profile", help="Profile of the loaded data")
    add_common_args(profile_parser)

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
        "processor"     : test_preprocessing,
        "load"          : test_load,
        "shuffle-load"  : test_load_shuffled,
        "summary"        : test_load_summary,
        "profile"       : test_load_profile
    }

    try:
        logger.info(f"Starting test: {args.command}")
        commands[args.command](args, logger)
        logger.info(f"Test completed: {args.command}")
    except Exception as e:
        logger.critical(f"Test failed: {e}")
        raise
    finally: Logger.shutdown_all()


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
