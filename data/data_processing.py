"""
    data/data_processing.py
    -----------------------
    Processing the data from the stored data files, this script manages the 
    extraction from these files into numpy string dictionary representation. 
"""
import orjson
import numpy as np
import pandas as pd


from itertools import chain
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from logs.logger import Logger


class DataProcessor:
    """
    Handles structured transformation of input tabular data (e.g., DataFrame
    chunks) into consistent NumPy representations according to a predefined
    schema.

    Each schema entry defines:
        - Expected NumPy dtype
        - Whether the column represents a stacked (nested) structure

    Gracefully handles:
        - Missing columns
        - Malformed JSON
        - Ragged lists
        - Large dataset decoding (thread-parallel JSON parsing)
    """
    SCHEMA = {
        "dvec"      : ( np.float32,  True  ),
        "rx_type"   : ( np.uint8,    False ),
        "link_state": ( np.uint8,    False ),
        "los_pl"    : ( np.float32,  False ),
        "los_ang"   : ( np.float32,  True  ),
        "los_dly"   : ( np.float32,  False ),
        "nlos_pl"   : ( np.float32,  True  ),
        "nlos_ang"  : ( np.float32,  True  ),
        "nlos_dly"  : ( np.float32,  True  )
    }
    JSON_THREAD_THRESHOLD = 100_000

    def __init__(self, logger: Logger): self.logger = logger


    # ---------------========== Chunk Processing ==========--------------- #

    def process_chunk(self, chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert a single DataFrame chunk into typed NumPy arrays following
        the schema rules.

        Returns:
        --------
            dict[str, np.ndarray]: Processed column data
        """
        processed: Dict[str, np.ndarray] = {}
        for column, (dtype, stacked) in self.SCHEMA.items():
            if column not in chunk.columns: continue

            values = chunk[column].to_numpy()
            if len(values) == 0:
                processed[column] = np.empty((0,), dtype=dtype)
                continue

            # ---------------- Numeric / simple columns ---------------- #
            if not stacked or values.dtype != object:
                processed[column] = self._convert_simple_column(values, dtype, column)
                continue

            # ---------------- Stacked / JSON columns ---------------- #
            processed[column] = self._convert_stacked_column(values, dtype, column)

        return processed


    # ---------------========== Simple Numeric Columns ==========--------------- #

    def _convert_simple_column(self, values: np.ndarray, dtype, col: str) -> np.ndarray:
        try: return values.astype(dtype, copy=False)
        except Exception:
            self.logger.debug(f"Column `{col}` contains non-castable values; using dtype=object.")
            return np.array(values, dtype=object)


    # ---------------========== Stacked / JSON Columns ==========--------------- #

    def _convert_stacked_column(self, values: np.ndarray, dtype, col: str) -> np.ndarray:
        first_val = values[0]

        # JSON strings
        if isinstance(first_val, str):
            return self._decode_json_array(values, dtype, col)

        # Already lists/arrays
        try:
            return np.array(values.tolist(), dtype=dtype)
        except Exception:
            self.logger.debug(f"Column `{col}` stored as ragged arrays; falling back to object.")
            return np.array(values, dtype=object)


    def _decode_json_array(self, values: np.ndarray, dtype, col: str) -> np.ndarray:
        """
        Parallel / sequential JSON decoding based on data size.
        """
        try:
            if len(values) > self.JSON_THREAD_THRESHOLD:
                with ThreadPoolExecutor() as tpe:
                    decoded = list(tpe.map(orjson.loads, values))
            else:
                decoded = [orjson.loads(v) for v in values]

            return np.array(decoded, dtype=dtype)

        except Exception:
            self.logger.warning(f"Column `{col}` contains malformed JSON; coercing to dtype=object.")
            return np.array([self._safe_parse(v) for v in values], dtype=object)


    # ---------------========== Concatenating Many Chunks ==========--------------- #

    def concatenate_results(self, results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Combine multiple processed chunk dictionaries.

        Handles:
            - Numeric arrays (np.concatenate)
            - Ragged arrays (converted to dtype=object)
        """
        if not results:
            return {key: np.empty((0,), dtype=dtype) for key, (dtype, _) in self.SCHEMA.items()}

        output: Dict[str, np.ndarray] = {}
        for key in results[0].keys():
            arrays = [result[key] for result in results if key in result]
            if not arrays: continue

            try: output[key] = np.concatenate(arrays, axis=0)
            except Exception:
                # Ragged fallback
                flattened = list(chain.from_iterable((
                    array.tolist() if not isinstance(array, list) else array
                ) for array in arrays))
                output[key] = np.array(flattened, dtype=object)

        return output


    # ---------------========== JSON Safe Parsing ==========--------------- #

    @staticmethod
    def _safe_parse(value: Any) -> Any:
        """Attempt to parse JSON; return None on any failure."""
        try: return orjson.loads(value)
        except Exception: return None