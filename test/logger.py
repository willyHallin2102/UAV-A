"""
    debug / logger.py
    -----------------
    Logger instance CLI script for debugging functionality of features
    of the logger class.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
import random
import threading
import concurrent.futures

from typing import Any, Dict, List, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from debug.argparser import build_parser, CommandSpec
from debug.mainrunner import mainrunner
from logs.logger import Logger, LogLevel, get_loglevel



# ============================================================
#       Debugging Testing Methods 
# ============================================================


def test_loglevels(args: argparse.Namespace, logger: Logger):
    """ Testing the different log-levels """
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")


def test_extra_fields(args: argparse.Namespace, logger: Logger):
    """ Testing structured logging with additional extra fields """
    logger.info("User Action", user_name="abc", action="some action")
    logger.warning("Resource usage", cpu_percentage="100%!!", memory="2048 MB")

    # Nested-like structures
    logger.info("Database Query", query="SELECT * FROM users WHERE active=TRUE")

    logger.error("API Call failed", response_time="4days")

    config = {
        "max_connections": 100,
        "timeout_seconds": 30,
        "retry_policy": {"max_retries": 3, "backoff": "exponential"},
        "features": ["compression", "encryption", "caching"]
    }
    logger.debug("Application configuration", config=config)


def test_exception_logging(args: argparse.Namespace, logger: Logger):
    """ Test exception logging with traceback """
    # Try log an exception
    try:
        if random.choice([True, False]):
            result = 10 / 0
        else:
            with open("nonexistent_file.txt", "r") as f:
                content = f.read()
    except (ZeroDivisionError, FileNotFoundError) as e:
        logger.exception(
            "An error occurred during operation",
            operation_type="data-processing",
            stage="Initialization",
            error_type=type(e).__name__
        )
    
    def risky_operation():
        try:
            data = {"a": 1, "b": 2}
            return data["c"]
        except KeyError as e:
            raise ValueError("Missing required key") from e
    
    try: risky_operation()
    except Exception as e:
        logger.exception(
            "Chained exception occurred",
            module=__name__, function="test_exception_logging"
        )


def test_timed_operations(args: argparse.Namespace, logger: Logger):
    """ Test the timed context manager """
    print("\n=== Testing Timed Operations ===")
    
    # Basic timing
    with logger.timed("database_query"):
        time.sleep(0.2)
    
    # Nested timing
    with logger.timed("complex_operation"):
        time.sleep(0.1)
        with logger.timed("sub_operation_1", level=LogLevel.DEBUG):
            time.sleep(0.05)
        with logger.timed("sub_operation_2", level=LogLevel.DEBUG):
            time.sleep(0.07)
    
    # Timing with custom level
    with logger.timed("background_task", level=LogLevel.INFO):
        # Simulate work with progress updates
        for i in range(3):
            logger.debug(f"Processing item {i+1}/3")
            time.sleep(0.05)
    
    # Multiple concurrent timings
    operations = ["data_fetch", "data_transform", "data_store"]
    for op in operations:
        with logger.timed(f"pipeline_{op}"):
            time.sleep(random.uniform(0.05, 0.15))


def test_concurrent_logging(args: argparse.Namespace, logger: Logger):
    """ Test logging from multiple threads """
    def worker_thread(worker_id: int, iterations: int = 5):
        """ Simulate a worker thread that logs messages """
        thread_logger = Logger.get_logger(f"worker.{worker_id}", LogLevel.DEBUG)
        for i in range(iterations):
            # Simulate some work
            time.sleep(random.uniform(0.01, 0.05))

            # Log with thread-specific info
            thread_logger.info(
                f"Worker {worker_id} processing task {i+1}", worker_id=worker_id,
                task_number=i+1, thread_name=threading.current_thread().name
            )

            # Occasionally log errors
            if random.random() < 0.2:
                thread_logger.warning(
                    f"Minor issue in worker {worker_id}",
                    worker_id=worker_id, issue_type="resource_wait"
                )
    
    # Create and run threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker_thread, args=(i, 3))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads: t.join()
    logger.info("All worker threads completed successfully", total_workers=5)


def test_performance(args: argparse.Namespace, logger: Logger):
    """ Test logger performance under load """
    num_messages = args.num_messages if hasattr(args, 'num_messages') else 1000
    batch_size = 100
    logger.info(f"Starting performance test with {num_messages} messages")
    start_time = time.perf_counter()
    
    # Use ThreadPoolExecutor for concurrent logging
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        def log_batch(batch_id: int, messages_in_batch: int):
            for i in range(messages_in_batch):
                level = random.choice(["DEBUG", "INFO", "WARNING"])
                msg = f"Performance test message {batch_id}-{i}"
                
                if level == "DEBUG":
                    logger.debug(msg, batch=batch_id, msg_num=i)
                elif level == "INFO":
                    logger.info(msg, batch=batch_id, msg_num=i)
                elif level == "WARNING":
                    logger.warning(msg, batch=batch_id, msg_num=i)
        
        # Submit batches
        for batch in range(0, num_messages, batch_size):
            batch_id = batch // batch_size
            remaining = min(batch_size, num_messages - batch)
            futures.append(executor.submit(log_batch, batch_id, remaining))
        
        # Wait for completion
        concurrent.futures.wait(futures)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    msg_per_sec = num_messages / total_time if total_time > 0 else 0
    
    logger.info(
        "Performance test completed", total_messages=num_messages,
        total_time_sec=round(total_time, 3), messages_per_sec=round(msg_per_sec, 1),
        avg_latency_ms=round((total_time / num_messages) * 1000, 3)
    )


def test_logger_lifecycle(args: argparse.Namespace, logger: Logger):
    """ Test logger initialization, configuration changes, and shutdown """
    # Test getting multiple loggers
    loggers = []
    for i in range(3):
        name = f"module.component.{i}"
        component_logger = Logger.get_logger(name, LogLevel.DEBUG if i == 0 else LogLevel.INFO)
        loggers.append(component_logger)
        component_logger.info(f"Component {i} initialized", component_id=i)
    
    # Simulate different logging patterns
    logger.debug("Debug message from root logger")
    logger.info(
        "Info message with structured data", 
        components_active=len(loggers), test_phase="lifecycle"
    )
    
    # Test logging after some time (simulating long-running app)
    time.sleep(0.1)
    logger.warning(
        "Application warning - resource threshold exceeded",
        memory_usage_percent=92.5, cpu_usage_percent=88.3
    )
    
    # Test error scenario
    try:
        raise ConnectionError("Failed to connect to database server")
    except ConnectionError as e:
        logger.exception(
            "Critical infrastructure error", service="database",
            retry_attempt=3, timeout_seconds=30
        )
    logger.info("Lifecycle test completed successfully")


def test_file_rotation(args: argparse.Namespace, logger: Logger):
    """ Test file rotation by generating enough logs to trigger rotation """
    # This test requires --to-disk flag to be set
    if not args.to_disk:
        logger.warning("File rotation test requires --to-disk flag")
        print("Skipping - run with: --to-disk")
        return
    
    messages_per_batch = 50  # Reduced for testing
    estimated_bytes_per_message = 200  # Rough estimate
    logger.info("Starting file rotation test")
    
    # Generate structured logs with varying sizes
    for batch in range(3):
        for i in range(messages_per_batch):
            # Create messages with different sizes
            message_size = random.randint(50, 500)
            message = "X" * message_size
            logger.info(
                f"Rotation test batch {batch} message {i}", batch_number=batch,
                message_index=i, message_size=message_size, timestamp=datetime.now().isoformat(),
                random_data={
                    "value": random.random(),
                    "items": [random.randint(1, 100) for _ in range(5)],
                    "nested": {"level": random.randint(1, 3)}
                }
            )
        logger.info(f"Completed batch {batch} of rotation test")
    logger.info("File rotation test completed - check log files for rotation")

# ============================================================
#       Main Script
# ============================================================

@mainrunner
def main():
    parser = build_parser([
        CommandSpec(name="level",help="Test logging levels",handler=test_loglevels),
        CommandSpec(name="extra",help="Test extra fields",handler=test_extra_fields),
        CommandSpec(name="exception",help="Test Logger Exception",handler=test_exception_logging),
        CommandSpec(name="time", help="Testing timer", handler=test_timed_operations),
        CommandSpec(name="concurrent",help="Test concurrency",handler=test_concurrent_logging),
        CommandSpec(name="performance",help="Testing logger performance",handler=test_performance),
        CommandSpec(name="lifecycle",help="Testing the Logger lifecycle",handler=test_logger_lifecycle),
        CommandSpec(name="rotation",help="testing the logger file-rotation",handler=test_file_rotation)
    ])
    args = parser.parse_args()

    Logger.configure(
        log_directory="logger-debug", use_console=True, use_json=True, to_disk=True
    )
    logger = Logger.get_logger("debug", LogLevel[args.loglevel])

    logger.info(f"Starting test: {args.command}")
    args._handler(args, logger)
    logger.info(f"Completed test: {args.command}")


if __name__ == "__main__": 
    main()
