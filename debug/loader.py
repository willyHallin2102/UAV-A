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
import matplotlib.pyplot as plt

from logs.logger import Logger, LogLevel, get_loglevel
from data.get_data import get_city_data, get_shuffled_city_data
from data.loader import DataLoader, shuffle_and_split
from data.data_processing import DataProcessor
from data.file_handler import HandlerFactory, CsvHandler



def load(args: argparse.ArgumentParser, logger: Logger):
    """Test basic data loading functionality"""
    logger.info(f"Loading data for city: {args.city}")
    with logger.time_block(f"Loading {args.city} data", level=LogLevel.INFO):
        data = get_city_data(args.city, logger=logger)
    logger.info(
        f"Loaded {len(data)} columns with shapes: "
        f"{{ {', '.join(f'{k}: {v.shape}' for k, v in data.items())} }}"
    )
    
    if args.sample:
        logger.info("Sample data:")
        for key, array in data.items():
            logger.info(f"  {key}: {array[:2] if len(array) > 0 else 'empty'}")


def load_shuffled(args: argparse.ArgumentParser, logger: Logger):
    """Test shuffled data loading with train/validation split"""
    logger.info(f"Loading shuffled data for city: {args.city}")
    with logger.time_block(f"Loading shuffled {args.city} data", level=LogLevel.INFO):
        train_data, val_data = get_shuffled_city_data(
            args.city, logger=logger, validation_ratio=args.val_ratio
        )
    
    logger.info(f"Train data: {len(train_data)} columns, shapes: {{ {', '.join(f'{k}: {v.shape}' for k, v in train_data.items())} }}")
    logger.info(f"Validation data: {len(val_data)} columns, shapes: {{ {', '.join(f'{k}: {v.shape}' for k, v in val_data.items())} }}")


def test_processor(args: argparse.ArgumentParser, logger: Logger):
    """Test DataProcessor functionality directly"""
    logger.info("Testing DataProcessor with sample data...")
    
    # Create sample data that matches the schema
    sample_data = {
        'dvec': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        'rx_type': [1, 2, 1],
        'link_state': [0, 1, 0],
        'los_pl': [10.5, 11.2, 9.8],
        'los_ang': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        'los_dly': [1.1, 1.2, 1.3],
        'nlos_pl': [[20.1, 20.2], [21.1, 21.2], [22.1, 22.2]],
        'nlos_ang': [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
        'nlos_dly': [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]]
    }
    
    df = pd.DataFrame(sample_data)
    processor = DataProcessor(logger)
    
    with logger.time_block("DataProcessor chunk processing", level=LogLevel.INFO):
        processed = processor.process_chunk(df)
    
    logger.info("Processed data types:")
    for key, array in processed.items():
        logger.info(f"  {key}: {array.dtype} - shape: {array.shape} - sample: {array[:1]}")


def test_file_handler(args: argparse.ArgumentParser, logger: Logger):
    """Test file handler functionality"""
    logger.info("Testing file handler functionality...")
    
    # Get actual city data to test with
    data = get_city_data(args.city, logger=logger)
    loader = DataLoader(logger=logger)
    
    # Test saving
    test_filename = f"debug_test_{args.city}"
    with logger.time_block(f"Saving test data to {test_filename}", level=LogLevel.INFO):
        loader.save(data, test_filename, fmt="csv")
    
    # Test loading back
    with logger.time_block(f"Loading test data from {test_filename}", level=LogLevel.INFO):
        loaded_data = loader.load([f"{test_filename}.csv"])
    
    logger.info(f"Original vs loaded data comparison:")
    for key in data.keys():
        orig_shape = data[key].shape
        loaded_shape = loaded_data[key].shape
        logger.info(f"  {key}: {orig_shape} -> {loaded_shape}")
        
        if orig_shape == loaded_shape:
            logger.debug(f"    ✓ Shapes match")
        else:
            logger.warning(f"    ✗ Shape mismatch!")


def test_performance(args: argparse.ArgumentParser, logger: Logger):
    """Test performance with different configurations"""
    logger.info("Testing performance with different worker configurations...")
    
    cities = ["beijing", "boston"]  # Use smaller subset for performance testing
    
    for workers in [1, 2, 4]:
        for use_processes in [False, True]:
            logger.info(f"Testing with {workers} workers, processes={use_processes}")
            
            loader = DataLoader(
                n_workers=workers,
                prefer_processes=use_processes,
                logger=logger
            )
            
            with logger.time_block(f"Load performance test", level=LogLevel.INFO):
                try:
                    files = [f"uav_{city}/train.csv" for city in cities]
                    data = loader.load(files)
                    logger.info(f"  Success: loaded {len(data)} columns")
                except Exception as e:
                    logger.error(f"  Failed: {e}")


def test_schema_validation(args: argparse.ArgumentParser, logger: Logger):
    """Test schema validation and missing column handling"""
    logger.info("Testing schema validation...")
    
    # Test with incomplete data
    incomplete_data = {
        'rx_type': [1, 2, 1],
        'link_state': [0, 1, 0],
        'los_pl': [10.5, 11.2, 9.8],
        # Missing other required columns
    }
    
    df = pd.DataFrame(incomplete_data)
    processor = DataProcessor(logger)
    
    with logger.time_block("Processing incomplete data", level=LogLevel.INFO):
        processed = processor.process_chunk(df)
    
    logger.info("Results with incomplete data:")
    for key in processor.SCHEMA.keys():
        if key in processed:
            logger.info(f"  {key}: present - shape: {processed[key].shape}")
        else:
            logger.warning(f"  {key}: missing from processed data")


def analyze_data(args: argparse.ArgumentParser, logger: Logger):
    """Analyze loaded data statistics"""
    logger.info(f"Analyzing data for city: {args.city}")
    
    data = get_city_data(args.city, logger=logger)
    
    logger.info("Data Statistics:")
    for key, array in data.items():
        logger.info(f"  {key}:")
        logger.info(f"    dtype: {array.dtype}")
        logger.info(f"    shape: {array.shape}")
        
        if array.dtype in [np.float32, np.float64]:
            if len(array) > 0:
                logger.info(f"    min: {np.min(array):.4f}")
                logger.info(f"    max: {np.max(array):.4f}")
                logger.info(f"    mean: {np.mean(array):.4f}")
                logger.info(f"    std: {np.std(array):.4f}")
        elif array.dtype in [np.uint8, np.int32, np.int64]:
            if len(array) > 0:
                logger.info(f"    unique values: {np.unique(array)}")


def build_parser() -> argparse.ArgumentParser:
    """
    Forming the extended argument-parser
    """
    parser = argparse.ArgumentParser(description="Extended CLI tester for data loading and processing")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--loglevel", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
        p.add_argument("--city", type=str, default="beijing",
                       choices=["beijing", "boston", "london", "moscow", "tokyo"],
                       help="City to test with")

    # Load command
    load_parser = subparsers.add_parser("load", help='Test basic data loading')
    add_common(load_parser)
    load_parser.add_argument("--sample", action="store_true", 
                            help="Show sample data")

    # Load shuffled command
    shuffle_parser = subparsers.add_parser("load-shuffled", help='Test shuffled data loading')
    add_common(shuffle_parser)
    shuffle_parser.add_argument("--val-ratio", type=float, default=0.1,
                               help="Validation ratio for splitting")

    # Processor test command
    processor_parser = subparsers.add_parser("test-processor", help='Test DataProcessor directly')
    add_common(processor_parser)

    # File handler test command
    file_parser = subparsers.add_parser("test-file-handler", help='Test file handler functionality')
    add_common(file_parser)

    # Performance test command
    perf_parser = subparsers.add_parser("test-performance", help='Test performance with different configs')
    add_common(perf_parser)

    # Schema validation command
    schema_parser = subparsers.add_parser("test-schema", help='Test schema validation')
    add_common(schema_parser)

    # Data analysis command
    analysis_parser = subparsers.add_parser("analyze", help='Analyze data statistics')
    add_common(analysis_parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger(
        "loader-cli", to_disk=False, level=get_loglevel(args.loglevel)
    )

    try:
        if args.command == "load":
            load(args, logger)
        elif args.command == "load-shuffled":
            load_shuffled(args, logger)
        elif args.command == "test-processor":
            test_processor(args, logger)
        elif args.command == "test-file-handler":
            test_file_handler(args, logger)
        elif args.command == "test-performance":
            test_performance(args, logger)
        elif args.command == "test-schema":
            test_schema_validation(args, logger)
        elif args.command == "analyze":
            analyze_data(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        raise


if __name__ == "__main__":
    try: 
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
