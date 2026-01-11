"""
    tests / loader.py
    -.-.-.-.-.-.-.-.-
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import tempfile

import numpy as np
import pandas as pd

from data.loader import DataLoader, shuffle_and_split
from debug.argparser import CommandSpec, build_parser
from debug.mainrunner import mainrunner




def generate_test_data(n_rows: int=100) -> pd.DataFrame:
    """ Generate synthetic test-data that matches the DataProcessor.SCHEMA """
    return pd.DataFrame({
        'dvec'      : [json.dumps([np.random.randn(3).tolist()]) for _ in range(n_rows)],
        'rx_type'   : np.random.randint(0, 4, n_rows, dtype=np.uint8),
        'link_state': np.random.randint(0, 2, n_rows, dtype=np.uint8),
        'los_pl'    : np.random.randn(n_rows).astype(np.float32),
        'los_ang'   : [json.dumps(np.random.randn(4).tolist()) for _ in range(n_rows)],
        'los_dly'   : np.random.randn(n_rows).astype(np.float32),
        'nlos_pl'   : [json.dumps(np.random.randn(20).tolist()) for _ in range(n_rows)],
        'nlos_ang'  : [json.dumps(np.random.randn(20, 4).tolist()) for _ in range(n_rows)],
        'nlos_dly'  : [json.dumps(np.random.randn(20).tolist()) for _ in range(n_rows)]
    })

#   'dvec'      : [json.dumps([1, 2, 3]) if not (i & 1) else None for i in range(n_rows)]
def generate_malformed_data(n_rows: int=50) -> pd.DataFrame:
    """ Generate data with various issues for edge case testing """
    return pd.DataFrame({
        'dvec'      : [json.dumps([1, 2, 3]) if i % 2 == 0 else None for i in range(n_rows)],
        'rx_type'   : [i if i % 3 != 0 else None for i in range(n_rows)],
        'link_state': [str(i) for i in range(n_rows)],  # Strings instead of ints (could be ok)
        'los_pl'    : np.random.randn(n_rows).astype(np.float32),
        'los_ang'   : ['invalid json' if i % 5 == 0 else json.dumps([1,2,3,4]) for i in range(n_rows)],
    })


def create_test_csv(filepath: Path, n_rows: int=1000, malformed: bool=False):
    """Create test CSV in the datasets directory if path is relative"""
    if malformed: 
        df = generate_malformed_data(n_rows)
    else: 
        df = generate_test_data(n_rows)
    
    # If relative path, use datasets directory
    if not filepath.is_absolute():
        datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        filepath = datasets_dir / filepath
    
    df.to_csv(filepath, index=False)
    return filepath


def clean_test_files(args: argparse.Namespace):
    """Clean up test files from the datasets directory"""
    datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
    
    if not datasets_dir.exists():
        print(f"Datasets directory does not exist: {datasets_dir}")
        return
    
    # Remove test files (you could add a pattern like "test_*.csv")
    test_files = list(datasets_dir.glob("test_*.csv"))
    for file in test_files:
        try:
            file.unlink()
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Failed to remove {file}: {e}")
    
    print(f"Cleaned up {len(test_files)} test files")

# ============================================================
#       Debugging Testing Methods 
# ============================================================


def test_load(args: argparse.Namespace):
    loader = DataLoader()
    data = loader.load(args.dataset)


def test_simple_loading(args: argparse.Namespace):
    """ Test basic file loading """
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tf:
        filepath = Path(tf.name)
        create_test_csv(filepath, n_rows=args.rows, malformed=args.malformed)

        worker_count = [1, 2, 4, 8]
        for n_workers in worker_count:
            loader = DataLoader(n_workers=n_workers, chunk_size=100)
            data = loader.load(str(filepath))

            assert len(data['rx_type']) == args.rows, f"{n_workers} worker failed"
            print(f"{n_workers} worker(s): {args.rows} rows loaded")
    
    filepath.unlink()
    return True


def test_save_load_roundtrip(args: argparse.Namespace):
    """ Testing saving data and the load the saved data """
    # Create source file, large and small generated

    if args.samples > 10e+6: 
        print(f"Samples exceeded 1_000_000")
        return
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        path = Path(tf.name)
        create_test_csv(path, n_rows=args.samples)
    
    # Create output file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        output_path = Path(tf.name)
    
    try:
        loader = DataLoader()
        data = loader.load(str(path))

        # Save to output
        loader.save(data, output_path)
        print(f"Saved went okay")

        # load from output
        loader.load(str(output_path))
    
    finally:
        path.unlink()
        output_path.unlink()


@mainrunner
def main():
    parser = build_parser([
        CommandSpec(
            name="load", help="Test Loading dataset", handler=test_load, extra_args=[{
                "flags": ["dataset"], 
                "kwargs": {"type": str, "help": "Dataset file to load"}
            }]
        ),
        CommandSpec(
            name="basic", help="Test generate and load data", handler=test_simple_loading,
            extra_args=[{
                    "flags": ["--rows", "-r"], 
                    "kwargs": {"type": int, "default": 500, "help": "Number of rows to generate"}
                },
                {
                    "flags": ["--malformed", "-m"], 
                    "kwargs": {"action": "store_true","default":False,"help":"Generate malformed malformed data"}
                }]
        ),
        CommandSpec(
            name="saveload",help="Generate, save, and reload a dataset",handler=test_save_load_roundtrip,
            extra_args=[
                {
                    "flags": ["--samples", "-s"], 
                    "kwargs": {"type": int, "default": 1000, 
                              "help": "Number of samples to generate"}
                },
                {
                    "flags": ["--workers", "-w"], 
                    "kwargs": {"type": int, "default": None, 
                              "help": "Number of workers to use"}
                }
            ]
        ),
        CommandSpec(
            name="clean",
            help="Clean up test files from datasets directory",
            handler=lambda args: clean_test_files(args),
            extra_args=[]
        )
    ])
    args = parser.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
