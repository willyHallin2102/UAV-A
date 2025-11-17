"""
    data/file_handler.py
    --------------------
    Class for managing file processing, modular approach to enable additional 
    file-format support. 
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Iterator, List, Protocol, Type, Union
from abc import ABC, abstractmethod

import pyarrow as pa
import pyarrow.csv as pv
from pyarrow.lib import ArrowInvalid, ArrowTypeError

from logs.logger import Logger



# ---------------========== Base Interfaces ==========--------------- #

class BaseFileHandler(ABC):
    """
    Abstract base class defining shared API and logger support for 
    all classes inherent this api. 
    """
    def __init__(self, logger: Logger):
        self.logger = logger
    

    @abstractmethod
    def load_chunks(self, filepath: Path, chunk_size: int) -> List[pd.DataFrame]:
        """Load file into DataFrame chunks."""
        pass


    @abstractmethod
    def save(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
        """Save structured data to a file."""
        pass


    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Helper: Convert dict of numpy arrays to DataFrame.
        Converts object/nested arrays to list for serialization.
        """
        df_dict = {}
        for key, array in data.items():
            if array.dtype == object: df_dict[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in arr
                ]
            else: df_dict[key] = array
        return pd.DataFrame(df_dict)


# ---------------========== CSV Handler Implementation ==========--------------- #

class CsvHandler(BaseFileHandler):
    """Efficient chunked CSV handler using PyArrow."""

    def load_chunks(self, filepath: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file `{filepath}` not found")

        try:
            avg_row_size = 256
            block_size = max(1 << 20, chunk_size * avg_row_size)

            read_opts = pv.ReadOptions(block_size=block_size, use_threads=True)
            convert_opts = pv.ConvertOptions(auto_dict_encode=False)

            reader = pv.open_csv(
                filepath,
                read_options=read_opts,
                convert_options=convert_opts
            )

            for batch in reader:
                df = batch.to_pandas()
                if not df.empty: yield df

            self.logger.debug(f"[CsvHandler] Streamed CSV in chunks from: {filepath}")

        except Exception as e:
            self.logger.error(f"[CsvHandler] Failed to read `{filepath}`: {e}")
            raise


    def save(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
        df = self._prepare_dataframe(data)
        try:
            table = pa.Table.from_pandas(df, preserve_index=False)

            write_opts = pv.WriteOptions(include_header=True)
            pv.write_csv(table, filepath, write_options=write_opts)

            self.logger.info(f"[CsvHandler] Saved {len(df):,} rows to {filepath}")

        except (ArrowInvalid, ArrowTypeError) as e:
            self.logger.warning(f"[CsvHandler] Arrow write failed; using pandas: {e}")
            df.to_csv(filepath, index=False)
            self.logger.info(f"[CsvHandler] Saved {len(df):,} rows via pandas")

        except Exception as e:
            self.logger.error(f"[CsvHandler] Unexpected error: {e}")
            raise



# ---------------========== Handler Factory ==========--------------- #

class HandlerFactory:
    HANDLERS: Dict[str, Type[BaseFileHandler]] = {
        ".csv": CsvHandler,
        "csv": CsvHandler,
    }

    @classmethod
    def get_handler(cls, fmt_or_path: Union[str, Path], logger: Logger) -> BaseFileHandler:
        s = str(fmt_or_path).lower()
        key = Path(s).suffix or s
        handler_cls = cls.HANDLERS.get(key)
        if not handler_cls:
            raise ValueError(f"Unsupported format: `{key}`. Supported: {list(cls.HANDLERS.keys())}")
        return handler_cls(logger)

    @classmethod
    def register_handler(cls, fmt: str, handler_cls: Type[BaseFileHandler]) -> None:
        cls.HANDLERS[fmt.lower()] = handler_cls

