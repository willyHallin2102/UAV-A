"""
    data / loader.py
    ----------------
    Loader instance script including the main structure called into the program
    to load the datasets to train the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from data.processors import DataProcessor
from data.handlers import HandlerFactory



class DataLoader:
    """
    High-level loader of data for training datasets and enable all
    transformations from pd.DataFrame stored files ...
    """
    REQUIRED_COLUMNS = [
        'dvec', 'rx_type', 'link_state', 'los_pl',
        'los_ang', 'los_dly', 'nlos_pl', 'nlos_ang', 'nlos_dly'
    ]

    def __init__(self,
        n_workers: Optional[int]=None,chunk_size: int=10_000, 
        prefer_processes: bool=False
    ):
        """
            Initialize Data-Loader instance
        """
        self.directory = Path(__file__).parent / "datasets"
        self.directory.mkdir(parents=True, exist_ok=True) # Already exists, does nothing

        self.n_workers = n_workers or mp.cpu_count()
        self.chunk_size = max(100, chunk_size)
        self.prefer_processes = prefer_processes

        self.processor = DataProcessor()
    

    def load(self, filepaths: Union[str,List[str]]) -> Dict[str, np.ndarray]:
        """
        Load and process one or more datasets into structured NumPy arrays.

        Args: 
        -----
            filepaths: List or string of filepath where files are retrieved from
        
        Returns:
        --------
            Dictionary of the processed data.
        """
        if isinstance(filepaths,(str,Path)): filepaths = [filepaths]
        print(f"Loading `{len(filepaths)}` file(s) with `{self.n_workers}` worker(s)...")

        # Choose executor based on configuration
        exe = ProcessPoolExecutor if self.prefer_processes else ThreadPoolExecutor
        chunks = []
        for filepath in filepaths:
            path = Path(filepath) if Path(filepath).is_absolute() else self.directory / filepath
            if not path.exists():
                print(f"Error, `{path}` not found!")
                continue

            try: 
                handler = HandlerFactory.get_handler(path)

                # Process chunks in parallel
                with exe(max_workers=self.n_workers) as executor:
                    futures = []
                    for chunk in handler.load_chunks(path, self.chunk_size):
                        missing = [
                            column for column in self.REQUIRED_COLUMNS if column not in chunk.columns
                        ]
                        if missing:
                            print(f"Warning: Missing column in `{path}`: {missing}")
                            continue

                        future = executor.submit(self.processor.process_chunk, chunk)
                        futures.append(future)
                    
                    # Collect results
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=60) # 60 seconds timeout
                            if result: chunks.append(result)
                        
                        except Exception as e:
                            print(f"Chunk processing failed: {e}")
            
            except Exception as e:
                print(f"Failed to process `{path}`: {e}")
                continue
        
        if not chunks:
            raise RuntimeError("No data successfully were processed")
        
        # Concatenate all chunks
        processed = self.processor.concatenate(chunks)
        
        # Compute the number of rows
        rows = len(next(iter(processed.values()))) if processed else 0
        print(f"Successfully loaded `{rows}` rows from {len(filepaths)} file(s)")

        return processed
    

    def save(self,
        data: Dict[str,np.ndarray], filepath: Union[str,Path], fmt: Optional[str]=None
    ):
        """
        Save the processed data to a new file 

        Args:
        -----
            data: processed data dictionary
            filepath: Output file path
            fmt: File format (inferred from extension if None)
        """
        path = Path(filepath)
        if fmt is None:
            handler = HandlerFactory.get_handler(path)
        else:
            if fmt not in HandlerFactory.HANDLERS:
                raise ValueError(f"Unsupported format: `{fmt}`")
            handler_class = HandlerFactory.HANDLERS[f".{fmt}"]
            handler = handler_class()
        
        handler.save(data, path)
    

    def validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Validate processed data consistency.
        
        Args:
            data: Processed data dictionary
            
        Returns:
            True if data is valid
        """
        if not data: return False
        
        # Check all required columns are present
        missing = [column for column in self.REQUIRED_COLUMNS if column not in data]
        if missing:
            print(f"Missing required columns: {missing}")
            return False
        
        # Check consistent lengths
        lengths = {key: len(value) for key, value in data.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            print(f"Inconsistent array lengths: {lengths}")
            return False
        
        return True


def shuffle_and_split(
    data: Dict[str, np.ndarray], val_ratio: float = 0.2, 
    test_ratio: float = 0.0, seed: int = 42
) -> Tuple[Dict[str, np.ndarray], ...]:
    """
    Shuffle and split dataset into subsets.
    
    Args:
        data: Input data dictionary
        val_ratio: Validation set ratio (0-1)
        test_ratio: Test set ratio (0-1)
        seed: Random seed
        
    Returns:
        Tuple of split datasets (train, val, [test])
    """
    # Validate ratios
    if not 0 <= val_ratio <= 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if not 0 <= test_ratio <= 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError(f"Sum of val_ratio and test_ratio must be < 1")
    
    # Get consistent length
    lengths = {len(value) for value in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")
    
    n = next(iter(lengths))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    # Calculate split indices
    val_split = int(n * (1 - val_ratio - test_ratio))
    test_split = int(n * (1 - test_ratio)) if test_ratio > 0 else n
    
    train_idx = indices[:val_split]
    val_idx = indices[val_split:test_split]
    
    # Create splits
    train_data = {key: value[train_idx] for key, value in data.items()}
    val_data = {key: value[val_idx] for key, value in data.items()}
    
    if test_ratio > 0:
        test_idx = indices[test_split:]
        test_data = {key: value[test_idx] for key, value in data.items()}
        return train_data, val_data, test_data
    
    return train_data, val_data
