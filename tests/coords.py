"""
    tests / coords.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np

from src.maths import coords
from tests.debug.parser import build_parser, mainrunner, CommandSpec
from tests.debug.timer import Timer
from typing import Any, Dict, Tuple


def generate_cartesian_vectors(n: int, dim: int=3, seed: int=42) -> np.ndarray:
    np.random.seed(seed)
    if dim == 2:
        vectors = np.random.randn(n, 3).astype(np.float64)
        vectors[:,2] = 0.0

    elif dim == 3: 
        vectors = np.random.randn(n, 3).astype(np.float64)
    
    else: 
        raise ValueError(f"Unsupported dimensions: {dim}")
    
    return vectors


def generate_spherical_vectors(n: int, seed: int=42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    np.random.seed(seed)
    return (
        np.random.uniform(0.1,100.0,n).astype(np.float64),
        np.random.uniform(-180.0,180.0,n).astype(np.float64),
        np.random.uniform(0.0,180.0,n).astype(np.float64)
    )


# ============================================================
#       Debugging Testing Methods 
# ============================================================

def test_cartesian_to_spherical(args: argparse.Namespace):
    vectors = generate_cartesian_vectors(args.samples, args.dim, args.seed)
    with Timer("Cartesian to Spherical"):
        _, _, _ = coords.cartesian_to_spherical(vectors)


def test_spherical_to_cartesian(args: argparse.Namespace):
    r,p,t = generate_spherical_vectors(args.samples, args.seed)
    with Timer("Spherical to Cartesian"):
        _ = coords.spherical_to_cartesian(r,p,t)


def test_cartesian_to_spherical_roundtrip(args: argparse.Namespace):
    v1 = generate_cartesian_vectors(args.samples, args.dim, args.seed)
    with Timer("Cartesian to Spherical"):
        r,p,t = coords.cartesian_to_spherical(v1)
    
    with Timer("Spherical to Cartesian"):
        v2 = coords.spherical_to_cartesian(r,p,t)
    
    diff = v1 - v2
    print(f"V1 - V2 = {diff}")


def test_spherical_to_cartesian_roundtrip(args: argparse.Namespace):
    r1,p1,t1 = generate_spherical_vectors(args.samples, args.seed)
    with Timer("Spherical to Cartesian"):
        v = coords.spherical_to_cartesian(r1,p1,t1)
    
    with Timer("Cartesian to Spherical"):
        r2,p2,t2 = coords.cartesian_to_spherical(v)
    
    r, p, t = r2-r1, p2-p1, t2-t1
    print(f"r diff = {r}")
    print(f"phi diff = {p}")
    print(f"theta diff = {t}")


def test_add_angles(args: argparse.Namespace):
    pass


def test_sub_angles(args: argparse.Namespace):
    pass

# ============================================================
#       Main Script
# ============================================================

COMMON_ARGS = [
    {"flags": ["--samples","-n"], "kwargs": {"type": int, "default": 1000}},
    {"flags": ["--seed","-s"], "kwargs": {"type": int, "default": 42}},
    {"flags": ["--dim","-d"], "kwargs": {"type": int, "default": 3}},
]

@mainrunner
def main():
    p = build_parser([
        CommandSpec(
            name="cart2sph", help="conversion from cartesian vectors to spherical",
            handler=test_cartesian_to_spherical, args=[*COMMON_ARGS]
        ),
        CommandSpec(
            name="sph2cart", help="conversion from spherical parameters to cartesian",
            handler=test_cartesian_to_spherical, args=[*COMMON_ARGS]
        ),
        CommandSpec(
            name="cart2sph-round", 
            help="conversion from spherical parameters to cartesian",
            handler=test_cartesian_to_spherical_roundtrip, args=[*COMMON_ARGS]
        ),
        CommandSpec(
            name="sph2cart-round", 
            help="conversion from spherical parameters to cartesian",
            handler=test_spherical_to_cartesian_roundtrip, args=[*COMMON_ARGS]
        )
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
