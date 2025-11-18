"""
    debug/coords.py
    ---------------
    Testing mathematical operations, such as adding angles 
    as well as subtracting angles. Including conversation 
    between cartesian representation of vectors to spherical
    representations.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from logs.logger import Logger, LogLevel, get_loglevel
from maths import coords
from data.get_data import get_city_data


# ---------------========== Generate Vectors ==========--------------- #

def _generate_vectors(args, rng: np.random.Generator) -> np.ndarray:
    """
    Generate randomized vectors for random testing of the coords functions, in 
    additions a random number generator seed `rng` is added for reproducibility
    """
    v = rng.normal(size=(args.n_samples, 3)).astype(args.dtype)
    v /= np.linalg.norm(v, axis=1)[:, None]
    v *= rng.uniform(0.1, 10.0, size=(args.n_samples, 1))
    
    return v


def _generate_spherical_coordinates(
    args, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random spherical coordinates."""
    r = rng.uniform(0.1, 10.0, size=args.n_samples).astype(args.dtype)
    phi = rng.uniform(-180.0, 180.0, size=args.n_samples).astype(args.dtype)
    theta = rng.uniform(0.0, 180.0, size=args.n_samples).astype(args.dtype)

    return r, phi, theta


# ---------------========== Testing Functions ==========--------------- #

def test_cartesian_to_spherical(args, logger, rng):
    logger.info("Cartesian → spherical test started")
    v = _generate_vectors(args, rng)
    with logger.time_block("cartesian-to-spherical"):
        coords.cartesian_to_spherical(v)
    logger.info("Cartesian → spherical test completed")


def test_spherical_to_cartesian(args, logger, rng):
    logger.info("Spherical → cartesian test started")
    r, p, t = _generate_spherical_coordinates(args, rng)
    with logger.time_block("spherical-to-cartesian"):
        coords.spherical_to_cartesian(r, p, t)
    logger.info("Spherical → cartesian test completed")


def test_roundtrip(args, logger, rng):
    """Check roundtrip consistency between cartesian and spherical conversions."""
    if args.n_samples > 10_000:
        raise TimeoutError("Too many samples for roundtrip test.")

    logger.info("Roundtrip test started")
    errors = []

    for _ in range(args.trials):
        v1 = _generate_vectors(args, rng)
        r, phi, theta = coords.cartesian_to_spherical(v1)
        v2 = coords.spherical_to_cartesian(r, phi, theta)
        errors.append(np.linalg.norm(v2 - v1, axis=1))

    errors = np.concatenate(errors)
    logger.info(f"Max Error: {np.max(errors):.5e}\tMean Error: {np.mean(errors):.5e}")


def _generate_angle_pairs(args, rng):
    """Generate two sets of angles for rotation tests."""
    _, phi_0, theta_0 = _generate_spherical_coordinates(args, rng)
    _, phi_1, theta_1 = _generate_spherical_coordinates(args, rng)
    return phi_0, theta_0, phi_1, theta_1


def test_add_angles(args, logger, rng):
    phi_0, theta_0, phi_1, theta_1 = _generate_angle_pairs(args, rng)
    with logger.time_block("Add Angles"):
        coords.add_angles(phi_0, theta_0, phi_1, theta_1)
    logger.info("Add-angles test completed")


def test_sub_angles(args, logger, rng):
    phi_0, theta_0, phi_1, theta_1 = _generate_angle_pairs(args, rng)
    with logger.time_block("Sub Angles"):
        coords.sub_angles(phi_0, theta_0, phi_1, theta_1)
    logger.info("Sub-angles test completed")


def test_rotation(args, logger, rng):
    """Verify add_angles and sub_angles are inverses."""
    logger.info("Rotation consistency test started")
    errors = []

    with logger.time_block("Rotation Consistency"):
        for _ in range(args.trials):
            phi_0, theta_0, phi_1, theta_1 = _generate_angle_pairs(args, rng)
            phi_r, theta_r = coords.add_angles(phi_0, theta_0, phi_1, theta_1)
            phi_b, theta_b = coords.sub_angles(phi_r, theta_r, phi_1, theta_1)
            diff_phi = np.mod(phi_b - phi_0 + np.pi, 2 * np.pi) - np.pi
            diff_theta = theta_b - theta_0
            errors.append(np.sqrt(diff_phi**2 + diff_theta**2))

    errors = np.concatenate(errors)
    logger.info(f"Mean angular error: {np.mean(errors):.6e}, Max error: {np.max(errors):.6e}")




# ---------------========== CLI Parser ==========--------------- #

def build_parser() -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="Cli debug for coords.py")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_command(p: argparse.ArgumentParser):
        p.add_argument("--n-samples", type=int, default=1000)
        p.add_argument("--dtype", type=lambda x: getattr(np, x), default=np.float64)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--debug", action="store_true")
        p.add_argument("--trials", type=int, default=100)
    
    for cmd, help_text in [
        ("cart-to-sph", "Cartesian → spherical conversion"),
        ("sph-to-cart", "Spherical → cartesian conversion"),
        ("roundtrip", "Cartesian → spherical → cartesian error check"),
        ("add-angles", "Angle addition test"),
        ("sub-angles", "Angle subtraction test"),
        ("rotate", "Combined rotation consistency test"),
    ]:
        add_command(sub.add_parser(cmd, help=help_text))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger("coords-cli", to_disk=False, level=LogLevel.INFO)
    rng = np.random.default_rng(args.seed)
    
    commands = {
        "cart-to-sph": test_cartesian_to_spherical,
        "sph-to-cart": test_spherical_to_cartesian,
        "roundtrip": test_roundtrip,
        "add-angles": test_add_angles,
        "sub-angles": test_sub_angles,
        "rotate": test_rotation,
    }

    try: commands[args.command](args, logger, rng)
    finally: Logger.shutdown_all()



if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
