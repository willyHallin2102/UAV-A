
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Union

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from data.file_handler import HandlerFactory, BaseFileHandler
from data.data_processing import DataProcessor

from logs.logger import Logger, LogLevel


# ---------------========== Data Loader ==========--------------- #

class DataLoader:
    """High-level orchestrator for chunked data loading and transformation."""

    REQUIRED_COLUMNS: Final[List[str]] = [
        'dvec', 'rx_type', 'link_state', 'los_pl',
        'los_ang', 'los_dly', 'nlos_pl', 'nlos_ang', 'nlos_dly'
    ]

    def __init__(self,
        n_workers: Optional[int] = None, chunk_size: int = 10_000,
        level: LogLevel = LogLevel.INFO, use_console: bool = True,
        prefer_processes: bool = False, logger: Optional[Logger]=None
    ):
        """Initialize DataLoader."""
        self.directory = Path(__file__).parent / "datasets"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.n_workers = n_workers or mp.cpu_count()
        self.chunk_size = int(chunk_size)
        self.prefer_processes = prefer_processes

        self.logger = logger or Logger(
            "data-loader", level=level, json_format=True, use_console=use_console
        )
        self.processor = DataProcessor(self.logger)

    # -------------------- Saving -------------------- #

    def save(self,
        data: Dict[str, np.ndarray],
        filepath: Union[str, Path],
        fmt: str = "csv"
    ) -> None:
        """Save structured data using the appropriate handler."""
        filepath = Path(self.directory) / filepath
        filepath = filepath.with_suffix(f".{fmt.lower()}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        handler = HandlerFactory.get_handler(fmt, self.logger)
        handler.save(data, filepath)

    # -------------------- Loading -------------------- #

    def load(self, filepaths: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Load and process one or more datasets into structured NumPy arrays.
        Uses concurrent streaming for optimal throughput.
        """
        filepaths = [filepaths] if isinstance(filepaths, (str, Path)) else filepaths
        n_files = len(filepaths)
        self.logger.info(
            f"Loading {n_files} file(s) using up to {self.n_workers} worker(s)..."
        )

        # Choose executor type
        Executor = ProcessPoolExecutor if self.prefer_processes else ThreadPoolExecutor

        # Streaming results from multiple files
        chunk_results = []
        for filepath in filepaths:
            path = Path(self.directory) / filepath
            if not path.exists():
                self.logger.error(f"File not found: {path}")
                continue

            handler = HandlerFactory.get_handler(path, self.logger)

            # Stream chunks directly from handler
            with Executor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self.processor.process_chunk, chunk)
                    for chunk in handler.load_chunks(path, self.chunk_size)
                }
                for future in as_completed(futures):
                    try:
                        chunk_results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Chunk processing failed for {path}: {e}")

        if not chunk_results:
            raise RuntimeError("No data chunks were successfully processed.")

        processed = self.processor.concatenate_results(chunk_results)
        self.logger.info(
            f"Processed {len(processed)} columns from {n_files} file(s) successfully."
        )
        return processed


# ===================== Utility Functions ===================== #

def shuffle_and_split(
    data: Dict[str, np.ndarray],
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Shuffle and split dataset into training and validation subsets.
    """
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths detected: {lengths}")

    n = next(iter(lengths))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_idx = int(n * (1 - val_ratio))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return (
        {k: v[train_idx] for k, v in data.items()},
        {k: v[val_idx] for k, v in data.items()},
    )