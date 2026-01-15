"""
    data / handlers.py
    ------------------
    Script includes classes for a  variety of different file-managers designated
    for implementations for loading and saving different types of file formats. 
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



class FileHandler(ABC):
    """ Abstract base class of shared implementations """

    @abstractmethod
    def load_chunks(self, filepath: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """ Yield `pd.DataFrame` chunks from a file """
        raise NotImplementedError
    
    def save(self, data: Dict[str,np.ndarray], filepath: Path):
        """ Save processed data to file """
        if not data: raise ValueError("Cannot save empty data directory")

        df = self._prepare_dataframe(data)
        if df.empty: raise ValueError("Prepared dataframe is empty")
        
        self._write_dataframe(df, filepath)
        print(f"Saved data to `{filepath}`")
    
    @abstractmethod
    def _write_dataframe(self, df: pd.DataFrame, filepath: Path):
        """ Write `pd.DataFrame` to specific format """
        raise NotImplementedError
    

    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert processed array back to `pd.DataFrame`"""
        if not data: 
            return pd.DataFrame()
        
        df_dict = {}
        for key, array in data.items():
            array = np.asarray(array)
            
            if array.size == 0:
                df_dict[key] = []
                continue
            
            # Handle 1D numeric arrays efficiently
            if array.ndim == 1 and array.dtype != object:
                df_dict[key] = array.tolist()
                continue
            
            # Handle 2D+ arrays
            if array.ndim > 1:
                df_dict[key] = array.tolist()
                continue
            
            # Handle object arrays (the expensive case)
            # Pre-allocate list and use simple checks
            if array.dtype == object:
                column = [None] * len(array)
                for i, value in enumerate(array):
                    if isinstance(value, (list, np.ndarray, dict)):
                        # Only encode if necessary
                        column[i] = orjson.dumps(value).decode("utf-8")
                    else:
                        column[i] = str(value)
                df_dict[key] = column
            else:
                df_dict[key] = array.tolist()
        
        return pd.DataFrame(df_dict)


class CsvHandler(FileHandler):
    """ Chunked CSV handler using PyArrow """

    # Average row size in bytes (estimate)
    ROW_SIZE        : Final[int] = 1024
    MIN_BLOCK_SIZE  : Final[int] = 1 << 20    # 1MB

    def load_chunks(self, filepath: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """ Read CSV file in chunks using PyArrow """
        if not filepath.exists(): 
            raise FileNotFoundError(f"CSV file `{filepath}` not found")
        
        try:
            read_options = pv.ReadOptions(
                block_size=max(self.MIN_BLOCK_SIZE, chunk_size * self.ROW_SIZE),
                use_threads=True
            )
            convert_options = pv.ConvertOptions(
                auto_dict_encode=False, strings_can_be_null=True
            )
            reader = pv.open_csv(
                filepath, read_options=read_options, convert_options=convert_options
            )
            for batch in reader:
                df = batch.to_pandas()
                if not df.empty: yield df
            
            print(f"Loaded CSV from `{filepath}`")
        
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            print(f"Arrow error reading `{filepath}`: {e}")
            raise 

        except Exception as e:
            print(f"Error reading `{filepath}`: {e}")
            raise
    

    def _write_dataframe(self, df: pd.DataFrame, filepath: Path):
        """ Write `pd.DataFrame` to CSV """
        df.to_csv(filepath)




class HandlerFactory:
    """ Factory for creating appropriate file-handler """
    HANDLERS: Dict[str,Type[FileHandler]] = {
        ".csv": CsvHandler
    }

    @classmethod
    def get_handler(cls, path: Union[str,Path]) -> FileHandler:
        """ Get the appropriate file extension """
        path = Path(path)
        suffix = path.suffix.lower()
        if not suffix:
            raise ValueError(f"Cannot infer file type from file `{path}`")

        handler = cls.HANDLERS.get(suffix)
        if not handler:
            raise ValueError(
                f"!`{suffix}`, supported: `{', '.join(cls.HANDLERS.keys())}`"
            )
        return handler()
