"""
    data / processors.py
    -------------------------
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd

from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
from numpy.typing import DTypeLike



class DataProcessor:
    """
    Handles structured transformations of input tabular data (e.g.,
    pd.DataFrame chunks) into consistent NumPy arrays.
    """
    SCHEMA = {
        "dvec"      : {"dtype": np.float32, "stacked": True, "dim": 3},
        "rx_type"   : {"dtype": np.uint8,   "stacked": False},
        "link_state": {"dtype": np.uint8,   "stacked": False},
        "los_pl"    : {"dtype": np.float32, "stacked": False},
        "los_ang"   : {"dtype": np.float32, "stacked": True, "dim": 4},
        "los_dly"   : {"dtype": np.float32, "stacked": False},
        "nlos_pl"   : {"dtype": np.float32, "stacked": True, "dim": 20},
        "nlos_ang"  : {"dtype": np.float32, "stacked": True, "dim": (20, 4)},
        "nlos_dly"  : {"dtype": np.float32, "stacked": True, "dim": 20},
    }

    def __init__(self):
        """
            Initialize Data-Processor Instance
        """
        self._dtype_cache: Dict[str, np.dtype] = {}
    

    def process_chunk(self, chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert a single pandas DataFrame into typed NumPy arrays following the 
        self.SCHEMA structure

        Args:
        -----
            chunk: Input dataframe chunk
        """
        processed: Dict[str, np.ndarray] = {}
        for column, spec in self.SCHEMA.items():
            if column not in chunk.columns:
                continue
            
            dtype = spec["dtype"]
            stacked = spec.get("stacked", False)
            values = chunk[column].to_numpy(copy=False)

            if values.size == 0:
                processed[column] = np.empty((0,), dtype=dtype)
                continue

            try:
                if not stacked:
                    processed[column] = self._convert_simple_column(values, dtype, column)
                else:
                    processed[column] = self._convert_stacked_column(values, dtype, column)
            
            except Exception as e:
                print(f"Warning: Failed to process column: '{column}', dtype = object")
                processed[column] = np.asarray(values, dtype=object)
        
        return processed
    

    def _convert_simple_column(self,
        values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """ Convert non-stacked column """
        try: return values.astype(dtype, copy=False)
        except (ValueError, TypeError):
            print(f"Column `{column}` contains non-castable values; using dtype=object")
            return np.asarray(values, dtype=object)
    

    def _convert_stacked_column(self,
        values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """ Convert stacked columns with nested data """
        sample = self._first_valid(values)
        if sample is None: return np.asarray(values, dtype=object)
        
        # Handle JSON strings
        if isinstance(sample, str):
            return self._decode_json_array(values, dtype, column)
        
        # Handle already decoded lists/arrays
        try: return np.asarray(values.tolist(), dtype=dtype)
        except (ValueError, TypeError):
            print(f"Column `{column}` contains ragged arrays; fallback to object")
            return np.asarray(values, dtype=object)
    

    def _decode_json_array(self,
            values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """ Decode JSON arrays from string values """
        try: return np.asarray([
            orjson.loads(v) if v is not None else None for v in values
        ], dtype=dtype)
        except Exception:
            print(f"Column `{column}` contains malformed JSON; coercing to dtype=object")
            return np.array([self._safe_parse(v) for v in values], dtype=object)
    

    @staticmethod
    def _first_valid(values: np.ndarray) -> Any:
        """ Retrieve the first non-None value from passed array """
        for value in values:
            if value is not None:
                return value
        return None
    

    def concatenate(self,results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Concatenates multiple chunks results into a single directory.

        Args:
        -----
            results:    List of processed chunk directories
        
        Returns:
        --------
            Concatenated directory of arrays
        """
        if not results: return {
            key: np.empty((0,), dtype=spec["dtype"]) for key, spec in self.SCHEMA.items()
        }

        output: Dict[str, np.ndarray] = {}
        first_result = results[0]

        for key in first_result.keys():
            arrays = []
            for result in results:
                if key in result:
                    arrays.append(result[key])
            
            if not arrays:
                continue
            
            try:
                # Tries to concatenate with the original dtypes
                output[key] = np.concatenate(arrays, axis=0)
            except (ValueError, TypeError):
                # Fallback to object dtype concatenation
                print(f"Concatenate failed for `{key}`; using object dtype")
                obj_arrays = [np.asarray(a, dtype=object) for a in arrays]
                output[key] = np.concatenate(obj_arrays, axis=0)
        
        return output
    

    @staticmethod
    def _safe_parse(value: Any) -> Any:
        """ Safe parse JSON string """
        if not isinstance(value, str): return value
        try: return orjson.loads(value)
        except Exception: return None
