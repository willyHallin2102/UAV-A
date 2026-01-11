"""
    data / handlers.py
    ------------------
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Type, Union
from abc import ABC, abstractmethod


class Filehandler(ABC):
    """ Abstract base class for file-handlers """

    @abstractmethod
    def load_chunks(self, filepath: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """ Yield pd.DataFrame chunks from a file """
        raise NotImplementedError
    
    def save(self, data: Dict[str, np.ndarray], filepath: Path):
        """ Save processed data to file """
        if not data:
            raise ValueError("Cannot save empty data dictionary")
        
        df = self._prepare_dataframe(data)
        if df.empty:
            raise ValueError("Prepared dataframe is empty")
        
        self._write_dataframe(df, filepath)
        print(f"Saved data to '{filepath}'") 
    
    @abstractmethod
    def _write_dataframe(self, df: pd.DataFrame, filepath: Path):
        """ Write pd.DataFrame to specific format """
        raise NotImplementedError
    

    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """ Convert processed arrays back to pd.DataFrame """
        if not data: return pd.DataFrame()
        
        df: Dict[str, list] = {}
        for key, array in data.items():
            array = np.asarray(array)
        
        for key, array in data.items():
            array = np.asarray(array)
            
            if array.size == 0:
                df[key] = []
                continue
            
            if array.ndim > 1:
                df[key] = array.tolist()
                continue
            
            try:
                if array.dtype != object: df[key] = array.tolist()
                else:
                    column = []
                    for value in array:
                        if isinstance(value, (list, np.ndarray, dict)):
                            column.append(orjson.dumps(value).decode("utf-8"))
                        else: column.append(value)
                    
                    df[key] = column

            except Exception as e:
                print(f"Error processing {key}: {e}")
                df[key] = array.tolist()
        
        return pd.DataFrame(df)


class CsvHandler(Filehandler):
    """ Chunked CSV handler using PyArrow """

    # Average row size in bytes (estimate)
    ROW_SIZE = 1024
    MIN_BLOCK_SIZE = 1 << 20    # 1MB

    def load_chunks(self, filepath: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """ Read CSV file in chunks using PyArrow """
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file `{filepath}` not found")
        
        try:
            # Calculate blocksize based on chunk_size and row size estimate
            read_options = pv.ReadOptions(
                block_size=max(self.MIN_BLOCK_SIZE, chunk_size * self.ROW_SIZE),
                use_threads=True
            )
            convert_options = pv.ConvertOptions(
                auto_dict_encode=False, strings_can_be_null=True
            )

            # Open CSV with efficient streaming
            reader = pv.open_csv(
                filepath, read_options=read_options, convert_options=convert_options
            )

            for batch in reader:
                df = batch.to_pandas()
                if not df.empty: yield df
            
            # Logger
            print(f"Loaded CSV from '{filepath}'")
        
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            print(f"Arrow error reading `{filepath}`: {e}")
            raise
        
        except Exception as e:
            print(f"Error reading `{filepath}`: {e}")
            raise

    def _write_dataframe(self, df: pd.DataFrame, filepath: Path):
        """ Write pd.DataFrame to CSV """
        df.to_csv(filepath)





class HandlerFactory:
    """ Factory for creating appropriate file-handler """
    HANDLERS: Dict[str,Type[Filehandler]] = {
        ".csv": CsvHandler
    }

    @classmethod
    def get_handler(cls, path: Union[str,Path]) -> Filehandler:
        """ Get the appropriate file extension """
        path = Path(path)
        suffix = path.suffix.lower()

        if not suffix:
            raise ValueError(f"Cannot infer file type from file `{path}`")
        
        handler = cls.HANDLERS.get(suffix)
        if not handler:
            supported = ", ".join(cls.HANDLERS.keys())
            raise ValueError(
                f"Unsupported format `{suffix}`. supported: {supported}"
            )
        
        return handler()
    
    # Unnecessary perhaps, could manually add beside no Handler will exist
    @classmethod
    def register_handler(cls, fmt: str, handler: Type[Filehandler]):
        """ Register a new file handler type """
        if not fmt.startswith("."): fmt = f".{fmt}"
        cls.HANDLERS[fmt.lower()] = handler
