"""
    debug / stats.py
    ----------------
    Statistical analysis utilities.
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Results from comparing two arrays or functions."""
    mean_diff: float
    max_diff: float
    std_diff: float
    relative_error: float
    passed: bool
    tolerance: float
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ComparisonResult({status}): "
            f"mean={self.mean_diff:.2e} |  "
            f"max={self.max_diff:.2e} |  "
            f"rel_err={self.relative_error:.2e}"
        )


def compare_arrays(array1: np.ndarray, array2: np.ndarray, 
                   tolerance: float = 1e-12, 
                   relative: bool = False) -> ComparisonResult:
    """
    Compare two arrays numerically.
    
    Args:
        array1: First array
        array2: Second array
        tolerance: Absolute tolerance for comparison
        relative: If True, use relative tolerance
    
    Returns:
        ComparisonResult object
    """
    diff = np.abs(array1 - array2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # Calculate relative error
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    if norm1 > 0 and norm2 > 0:
        relative_error = np.linalg.norm(array1 - array2) / max(norm1, norm2)
    else: relative_error = 0.0 if max_diff == 0 else np.inf
    
    # Check if arrays are close
    if relative: passed = relative_error < tolerance
    else: passed = max_diff < tolerance
    
    return ComparisonResult(
        mean_diff=mean_diff, max_diff=max_diff, std_diff=std_diff, 
        relative_error=relative_error, passed=passed, tolerance=tolerance
    )


def roundtrip_error(
    original: np.ndarray, transformed: np.ndarray, tolerance: float = 1e-12
) -> ComparisonResult:
    """
    Calculate round-trip error.
    
    Args:
        original: Original array
        transformed: Array after transformation and inverse transformation
    
    Returns:
        ComparisonResult with error metrics
    """
    return compare_arrays(original, transformed, tolerance)


def print_comparison(
    results: Dict[str, ComparisonResult], title: str = "Comparison Results"
):
    """Print formatted comparison results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for name, result in results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{name}: {status}")
        print(f"  Mean difference:  {result.mean_diff:.2e}")
        print(f"  Max difference:   {result.max_diff:.2e}")
        print(f"  Std difference:   {result.std_diff:.2e}")
        print(f"  Relative error:   {result.relative_error:.2e}")
        print(f"  Tolerance:        {result.tolerance:.2e}")


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for an array."""
    return {
        'mean'  : float(np.mean(data)),
        'median': float(np.median(data)),
        'std'   : float(np.std(data)),
        'min'   : float(np.min(data)),
        'max'   : float(np.max(data)),
        'q1'    : float(np.percentile(data, 25)),
        'q3'    : float(np.percentile(data, 75)),
        'size'  : int(data.size),
    }


def print_statistics(data: np.ndarray, name: str = "Data"):
    """Print formatted statistics."""
    stats = calculate_statistics(data)
    
    print(f"\n{name} Statistics:")
    print(f"  Size:    {stats['size']:,}")
    print(f"  Mean:    {stats['mean']:.6f}")
    print(f"  Median:  {stats['median']:.6f}")
    print(f"  Std:     {stats['std']:.6f}")
    print(f"  Min:     {stats['min']:.6f}")
    print(f"  Max:     {stats['max']:.6f}")
    print(f"  Q1:      {stats['q1']:.6f}")
    print(f"  Q3:      {stats['q3']:.6f}")