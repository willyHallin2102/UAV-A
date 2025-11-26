"""
    data/file_handlers.py
    ---------------------
    Class `FileHandler` is a subclass for the `DataLoader` class
    for handling various file formats, this enable simpler
    extension for additional formats. The purpose of this class
    is to specifically reconfigure any given data-format stored 
    data into `pandas.DataFrame`
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


    # @abstractmethod
    # def save(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
    #     """Save structured data to a file."""
    #     pass

    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        df_dict: Dict[str, list] = {}
        for key, array in data.items():
            if isinstance(array, list): array = np.asarray(array, dtype=object)

            if array.ndim == 1:
                if array.dtype == object:
                    df_dict[key] = [orjson.dumps(value).decode("utf-8") if isinstance(
                            value, (list, np.ndarray)
                        ) else value for value in array]
                else: df_dict[key] = array.tolist()
            else: df_dict[key] = [
                orjson.dumps(value.tolist()).decode("utf-8") for value in array
            ]


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
                filepath, read_options=read_opts, convert_options=convert_opts
            )

            for batch in reader:
                df = batch.to_pandas()
                if not df.empty: yield df

            self.logger.debug(f"[CsvHandler] Streamed CSV in chunks from: {filepath}")

        except Exception as e:
            self.logger.error(f"[CsvHandler] Failed to read `{filepath}`: {e}")
            raise


    # def save(self, data, filepath):
    #     df = self._prepare_dataframe(data)

    #     try:
    #         # Detect columns that PyArrow cannot handle
    #         if any(df[col].dtype == object for col in df.columns):
    #             raise ArrowInvalid("Object columns — forcing pandas CSV write")

    #         table = pa.Table.from_pandas(df, preserve_index=False)
    #         write_opts = pv.WriteOptions(include_header=True)
    #         pv.write_csv(table, filepath, write_options=write_opts)

    #         self.logger.info(f"[CsvHandler] Saved {len(df):,} rows to {filepath}")

    #     except Exception as e:
    #         self.logger.warning(
    #             f"[CsvHandler] Arrow write failed; using pandas instead: {e}"
    #         )
    #         df.to_csv(filepath, index=False)
    #         self.logger.info(f"[CsvHandler] Saved {len(df):,} rows via pandas")




# ---------------========== Handler Factory ==========--------------- #

class HandlerFactory:
    HANDLERS: Dict[str, Type[BaseFileHandler]] = {
        ".csv": CsvHandler, "csv": CsvHandler,
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

