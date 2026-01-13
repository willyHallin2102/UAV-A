"""
    tests / coords.py
    -----------------
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np

from src.maths import coords
from debug.mainrunner import mainrunner
from debug.argparser import build_parser, CommandSpec
from debug.clock import FunctionBenchmark, Timer, benchmark_comparison
from debug.stats import compare_arrays, print_comparison, print_statistics
from typing import Any, Dict, Tuple


def generate_cartesian(n: int, dim: int=3, seed: int=42) -> np.ndarray:
    np.random.seed(seed)

    if dim == 2:
        vectors = np.random.randn(n, 3).astype(np.float64)
        vectors[:,2] = 0.0

    elif dim == 3: vectors = np.random.randn(n, 3).astype(np.float64)
    else: raise ValueError(f"Unsupported dimensions: {dim}")
    return vectors


def generate_spherical(n: int, seed: int=42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """ Generate random spherical coordinates """
    np.random.seed(seed)
    return (
        np.random.uniform(0.1, 100.0, n).astype(np.float64),
        np.random.uniform(-180, 180, n).astype(np.float64),
        np.random.uniform(0, 180, n).astype(np.float64)
    )


def benchmark(f, *args, repeat: int = 5, warmup: int = 2, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """
    Benchmark a function with warmup runs.
    
    Returns:
        (result, stats_dict) where stats_dict contains:
        - mean_time: Average execution time (seconds)
        - std_time: Standard deviation of execution time
        - min_time: Minimum execution time
        - max_time: Maximum execution time
        - speed: Operations per second (if n_samples provided)
    """
    # Warmup runs (discard results)
    for _ in range(warmup): f(*args, **kwargs)
    
    # Timed runs
    times, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    stats = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }
    return result, stats


def print_benchmark_results(stats: Dict[str, float], n_samples: int = None, label: str = ""):
    """Print formatted benchmark results."""
    if label:
        print(f"\n{'='*60}")
        print(f"Benchmark: {label}")
        print(f"{'='*60}")
    
    print(f"Time per call:")
    print(f"  Mean:  {stats['mean_time']*1e3:8.3f} ms")
    print(f"  Std:   {stats['std_time']*1e3:8.3f} ms")
    print(f"  Min:   {stats['min_time']*1e3:8.3f} ms")
    print(f"  Max:   {stats['max_time']*1e3:8.3f} ms")
    
    if n_samples:
        ops_per_sec = n_samples / stats['mean_time']
        print(f"\nPerformance:")
        print(f"  Operations/sec: {ops_per_sec:,.0f}")
        print(f"  Samples/sec:    {ops_per_sec:,.0f}")
    
    if 'times' in stats and len(stats['times']) > 1:
        print(f"\nIndividual runs:")
        for i, t in enumerate(stats['times'], 1):
            print(f"  Run {i}: {t*1e3:8.3f} ms")


# ============================================================
#       Debugging Testing Methods 
# ============================================================

def test_cartesian_to_cartesian(args: argparse.Namespace):
    vectors = generate_cartesian(args.samples, args.dim, args.seed)

    if args.samples == 1:
        with Timer("cartesian-to-spherical"):
            r, phi, theta = coords.cartesian_to_spherical(vectors[0])
            return
        
    benchmarker = FunctionBenchmark()
    stats = benchmarker.benchmark(
        coords.cartesian_to_spherical,args.repeat, args.warmup, vectors
    )
    print(f"\nPerformance:")
    print(f"  Mean time: {stats.mean*1000:.3f} ms")
    print(f"  Std time:  {stats.std*1000:.3f} ms")


def test_spherical_to_cartesian(args: argparse.Namespace):
    r, p, t = generate_spherical(args.samples, args.seed)
    results, stats = benchmark(
        coords.spherical_to_cartesian, r, p, t, repeat=args.repeat, warmup=2
    )
    print_benchmark_results(stats, args.samples, "spherical_to_cartesian")


def test_cartesian_to_spherical_roundtrip(args: argparse.Namespace):
    print(args.samples)
    v1 = generate_cartesian(args.samples, 3, args.seed)
    with Timer("Cartesian-to-Spherical: Roundtrip"):
        r, p, t = coords.cartesian_to_spherical(v1)
        v2 = coords.spherical_to_cartesian(r, p, t)
    
    error = v2 - v1
    print_statistics(error.flatten(), "Absolute Error")


def test_spherical_to_cartesian_roundtrip(args: argparse.Namespace):
    r1, p1, t1 = generate_spherical(args.samples, args.seed)
    with Timer("Cartesian-to-Spherical: Roundtrip"):
        v = coords.spherical_to_cartesian(r1, p1, t1)
        r2, p2, t2 = coords.spherical_to_cartesian(v)


def test_add_angles(args: argparse.Namespace):
    _, p1, t1 = generate_spherical(args.samples, seed=args.seed)
    _, p2, t2 = generate_spherical(args.samples, seed=args.seed)

    with Timer("Test-Add-Angles"):
        coords.add_angles(p1, t1, p2, t2)

def test_sub_angles(args: argparse.Namespace):
    _, p1, t1 = generate_spherical(args.samples, seed=args.seed)
    _, p2, t2 = generate_spherical(args.samples, seed=args.seed)

    with Timer("Test-Add-Angles"):
        coords.sub_angles(p1, t1, p2, t2)
    
# ============================================================
#       Main Function Building Main Script
# ============================================================

@mainrunner
def main():
    parser = build_parser([
        CommandSpec(
            name="cart2sph",
            help="Convert cartesian to spherical coordinates",
            handler=test_cartesian_to_cartesian,
            extra_args=[
                {"flags": ["--samples","-n"], "kwargs": {"type": int,"default":1000}},
                {"flags": ["--dim","-d"], "kwargs": {
                    "type": int, "default": 3,"choices": [2,3]
                },},
                {"flags": ["--repeat","-r"], "kwargs": {"type": int, "default": 5}},
                {"flags": ["--warmup","-w"], "kwargs": {"type": int, "default": 1}},
                {"flags": ["--seed","-s"], "kwargs": {"type": int, "default": 42}}
            ]
        ),
        CommandSpec(
            name="sph2cart",
            help="Convert spherical to cartesian coordinates",
            handler=test_spherical_to_cartesian,
            extra_args=[
                {"flags": ["--samples","-n"], "kwargs": {"type": int, "default": 1000}},
                {"flags": ["--repeat", "-r"], "kwargs": {"type": int, "default": 5}},
                {"flags": ["--seed", "-s"], "kwargs": {"type": int, "default": 42}},
            ]
        ),
        CommandSpec(
            name="cart2sph-roundtrip",
            help="Test round trip from cartesian to spherical to cartesian",
            handler=test_cartesian_to_spherical_roundtrip,
            extra_args=[
                {"flags": ["--samples","-n"], "kwargs": {"type": int, "default": 1000}},
                {"flags": ["--seed", "-s"], "kwargs": {"type": int, "default": 42}},
            ]   
        ),
        CommandSpec(
            name="sph2cart-roundtrip",
            help="Test round trip from spherical to cartesian back to spherical",
            handler=test_cartesian_to_spherical_roundtrip,
            extra_args=[
                {"flags": ["--samples","-n"], "kwargs": {"type": int, "default": 1000}},
                {"flags": ["--seed", "-s"], "kwargs": {"type": int, "default": 42}},
            ]   
        ),
        CommandSpec(
            name="add", help="Testing adding angles", handler=test_add_angles,
            extra_args=[
                {"flags": ["--samples","-n"], "kwargs": {"type": int, "default": 1000}},
                {"flags": ["--seed", "-s"], "kwargs": {"type": int, "default": 42}}
            ]
        ),
        CommandSpec(
            name="sub", help="Testing Subtracting angles", handler=test_sub_angles,
            extra_args=[
                {"flags": ["--samples","-n"], "kwargs": {"type": int, "default": 1000}},
                {"flags": ["--seed", "-s"], "kwargs": {"type": int, "default": 42}}
            ]
        )
    ])

    args = parser.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
