"""
    debug/logger.py
    ---------------
    Logger instance cli script for debugging the functionality its includes
    and enable a efficient fixable state approach.
"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
from typing import Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from logs.logger import Logger, LogLevel, get_loglevel

# Extra Failing

def test_basic(args: argparse.Namespace, logger: Logger):
    """ Testing the different log-level """
    ########################################
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")


def test_exception(args: argparse.Namespace, logger: Logger):
    """ Test the logger exception logging """
    try: raise ValueError("This is a test exception")
    except Exception as e: logger.exception("An exception occurred during test")


def test_timing(args: argparse.Namespace, logger: Logger):
    """ Testing timing context manager """
    with logger.time_block(label="test-operation"):
        time.sleep(0.1)
    
    with logger.time_block(label="critical-operation"):
        time.sleep(0.05)


def test_extra_fields(args: argparse.Namespace, logger: Logger):
    """ Testing logging with some extra fields """
    logger.info("User action", user_id=123, action="login", ip="192.168.1.1")
    logger.error("API call failed", endpoint="/api/users", status_code=500, method="POST")


def test_performance(args: argparse.Namespace, logger: Logger):
    """ Test logging performance """
    start = time.time()
    count = 1000
    
    for i in range(count):
        logger.debug(f"Performance test message {i}")
    
    duration = time.time() - start
    logger.info(f"Logged {count} messages in {duration:.3f} seconds")
    logger.info(f"Rate: {count/duration:.1f} messages/second")



def test_file_rotation(args: argparse.Namespace, logger: Logger):
    """ Test file rotation by generating large logs """
    large_message = "X" * 5000  # 5KB per message
    
    for i in range(100):  # Generate ~500KB of logs
        logger.info(f"Rotation test {i}: {large_message}")


def test_multiple_loggers(args: argparse.Namespace, logger: Logger):
    """ Test multiple logger instances """
    logger1 = Logger.get_logger("module.auth", to_disk=args.to_disk)
    logger2 = Logger.get_logger("module.api", to_disk=args.to_disk)
    logger3 = Logger.get_logger("module.db", to_disk=args.to_disk)
    
    logger1.info("Authentication module message")
    logger2.info("API module message", endpoint="/users")
    logger3.info("Database module message", query="SELECT * FROM users")


def test_log_levels(args: argparse.Namespace, logger: Logger):
    """Test different log levels filtering"""
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    for level_name in levels:
        temp_logger = Logger.get_logger(
            f"test_{level_name.lower()}",
            level=get_loglevel(level_name),
            to_disk=args.to_disk
        )
        temp_logger.debug(f"DEBUG message at {level_name} level")
        temp_logger.info(f"INFO message at {level_name} level")
        temp_logger.warning(f"WARNING message at {level_name} level")
        temp_logger.error(f"ERROR message at {level_name} level")
        temp_logger.critical(f"CRITICAL message at {level_name} level")


def test_stress(args: argparse.Namespace, logger: Logger):
    """ Stress test the logger """
    def stress_worker(worker_id, messages):
        for i in range(messages):
            logger.info(f"Stress worker {worker_id} message {i}")
    
    workers = 5
    messages_per_worker = 200
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(stress_worker, i, messages_per_worker)
            for i in range(workers)
        ]
        for future in futures:
            future.result()
    
    logger.info(
        f"Stress test completed: {workers * messages_per_worker} total messages"
    )


# ---------------========== Build Argument Parser ==========--------------- #

def build_parser() -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="CLI tester for link state predictor")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    def add_common_args(p: argparse.ArgumentParser):
        p.add_argument(
            "--loglevel", type=str, default="INFO", help="loglevel assignment",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        p.add_argument(
            "--to-disk", action="store_true",
            help="Enable disk logging (default: console only)"
        )
        p.add_argument(
            "--no-json", action="store_true",
            help="Disable JSON formatting (use console formatting)"
        )
    
    basic_parser = subparsers.add_parser("basic", help="Test basic logging functionality")
    add_common_args(basic_parser)

    exception_parser = subparsers.add_parser("exception", help="Test exception logging")
    add_common_args(exception_parser)

    timing_parser = subparsers.add_parser("timing", help="Test timing context manager")
    add_common_args(timing_parser)

    extra_parser = subparsers.add_parser("extra", help="Test extra fields")
    add_common_args(extra_parser)

    perf_parser = subparsers.add_parser("performance", help="Test logging performance")
    add_common_args(perf_parser)

    rotation_parser = subparsers.add_parser("rotation", help="Test file rotation")
    add_common_args(rotation_parser)
    rotation_parser.add_argument("--message-size", type=int, default=5000, help="Size of each message")

    multi_parser = subparsers.add_parser("multiple", help="Test multiple logger instances")
    add_common_args(multi_parser)

    levels_parser = subparsers.add_parser("levels", help="Test different log levels")
    add_common_args(levels_parser)

    stress_parser = subparsers.add_parser("stress", help="Stress test the logger")
    add_common_args(stress_parser)
    stress_parser.add_argument("--workers", type=int, default=5, help="Number of worker threads")
    stress_parser.add_argument("--messages", type=int, default=200, help="Messages per worker")

    all_parser = subparsers.add_parser("all", help="Run all tests")
    add_common_args(all_parser)

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
        "basic"         : test_basic,
        "exception"     : test_exception,
        "timing"        : test_timing,
        "extra"         : test_extra_fields,
        "performance"   : test_performance,
        "rotation"      : test_file_rotation,
        "multiple"      : test_multiple_loggers,    # Dead lock I think
        "levels"        : test_log_levels,          # Doesn't run 
        "stress"        : test_stress,              # mm 0 1 seem to stuck the rest
        "all"   : lambda args, logger: run_all_tests(args, logger)
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


# ---------------========== Testing all scripts ==========--------------- #


def run_all_tests(args: argparse.Namespace, logger: Logger):
    """Run all tests in sequence"""
    tests = [
        test_basic, test_exception, test_timing, test_extra_fields,
        test_multiple_loggers, test_performance,
    ]
    
    for f in tests:
        logger.info(f"Running test: {f.__name__}")
        try:
            test_func(args, logger)
            logger.info(f"✓ {f.__name__} passed")
        except Exception as e:
            logger.error(f"✗ {f.__name__} failed: {e}")
        time.sleep(0.1)  # Small delay between tests, useful makes it easier



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
