"""
    tests / loader.py
    -----------------
    CLI Script for managing 
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from data.loader import DataLoader, shuffle_and_split
from tests.debug.parser import CommandSpec, build_parser, mainrunner
from tests.debug.timer import Timer


# ============================================================
#       Debugging Testing Methods 
# ============================================================

def test_load_data(args: argparse.Namespace):
    """ Testing loading dataset """
    times: List[float] = []
    loader = DataLoader()
    for _ in range(args.repeats):
        with Timer() as t:
            data = loader.load([args.dataset])
        times.append(t.time_ms)
    
    avg = sum(times) / len(times)
    print(f"Average load time: {avg:.3f} ms")
    print(f"Min: {min(times):.3f} ms, Max: {max(times):.3f} ms")


def test_data_shape(args: argparse.Namespace):
    loader = DataLoader()
    data = loader.load([args.dataset])

    first_key = list(data.keys())[0] if data else None
    rows = len(data[first_key]) if first_key else 0
    columns = len(data)
    
    print(f"Data shape: {rows} rows × {columns} columns")
    print(f"Required columns ({len(loader.REQUIRED_COLUMNS)}): {loader.REQUIRED_COLUMNS}")
    print(f"Loaded columns ({columns}): {list(data.keys())}")

    print("\nData types:")
    for key, array in data.items():
        print(f"  {key}: {array.dtype}, shape: {array.shape}")


def test_concurrent_loading(args: argparse.Namespace):
    times = []

    for n_workers in [1, 2, 4, 8]:
        loader = DataLoader(n_workers=n_workers, prefer_processes=args.proc)
        
        with Timer() as t: data = loader.load([args.dataset])

        row_count = len(data[list(data.keys())[0]]) if data else 0
        times.append((n_workers, t.time_ms, row_count))
    
    print(f"Concurrent loading performance ({args.dataset}):")
    print(f"{'Workers':<10} {'Time (ms)':<12} {'Rows':<12} {'Rows/sec':<12}")
    print("-" * 60)

    baseline_time = times[0][1] if times else 1
    for n_workers, time_ms, rows in times:
        rows_per_sec = (rows / (time_ms / 1000)) if time_ms > 0 else 0
        speedup = baseline_time / time_ms if time_ms > 0 else 0
        print(f"{n_workers:<10} {time_ms:<12.2f} {rows:<12,} {rows_per_sec:<12,.0f} (×{speedup:.2f})")

# ============================================================
#       Main Script
# ============================================================

DATASET = [
    {"flags": ["dataset"], "kwargs": {"type": str, "default": "uav_london/train.csv"}},
    {"flags": ["--repeats","-r"], "kwargs": {"type": int, "default": 1}},
    {"flags": ["--seed","-s"], "kwargs": {"type": int, "default": 42}},
    {"flags": ["--proc"], "kwargs": {"action": "store_true"}},
    {"flags": ["--chunk-size","-cs"], "kwargs": {"type": int, "default": 10_000}}
]

@mainrunner
def main():
    p = build_parser([
        CommandSpec(
            name="load", help="Measure time of load the data",
            handler=test_load_data, args=[*DATASET]
        ),
        CommandSpec(
            name="data", help="Check dataset", handler=test_data_shape, args=[*DATASET]
        ),
        CommandSpec(
            name="concurrent", help="Testing various level of concurrency",
            handler=test_concurrent_loading, args=[*DATASET]
        )
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
