"""
    src / maths / coords.py
    -----------------------
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numba import njit, prange, vectorize, float64
from typing import Final, Optional, Tuple, Union

# Typing declarations
ArrayF = npt.NDArray[np.floating]
ArrayF64 = npt.NDArray[np.float64]

# Constants
DEG2RAD : Final[float] = np.pi / 180.0
RAD2DEG : Final[float] = 180.0 / np.pi
EPS     : Final[float] = 1e-12


# ----------======= Utilities =======---------- #

@njit(fastmath=True, cache=True)
def _as_1d_array_numba(x: np.ndarray) -> np.ndarray:
    """Numba-accelerated 1D array converter."""
    return np.array([x]) if x.ndim == 0 else x.ravel()


def _as_1d_array(x: Union[ArrayF, float]) -> ArrayF64:
    """ Convert input to a 1D NumPy array of dtype float64. """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 0:
        array = array[None]
    
    return array.ravel()


@vectorize([float64(float64)], nopython=True, cache=True)
def _clip_to_unit(x: float64) -> float64:
    """ Clip the z value """
    if x > 1.0:  return 1.0
    if x < -1.0: return -1.0
    return x

# ----------======== Coordinate Conversions ========---------- #

@njit(fastmath=True, parallel=True, cache=True)
def _cart2sph_kernel(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    r: np.ndarray, p: np.ndarray, t: np.ndarray
):
    """
    Numba kernel for cartesian to spherical conversion computes in place
    """
    n = len(x)
    for i in prange(n):
        xi, yi, zi = x[i], y[i], z[i]

        # Compute the radius
        ri = np.sqrt(xi * xi + yi * yi + zi * zi)
        if ri < EPS:
            r[i], p[i], t[i] = 0.0, 0.0, 0.0
            continue

        r[i], p[i] = ri, np.arctan2(yi, xi)
        ct = zi / ri
        if ct > 1.0     : ct = 1.0
        elif ct < -1.0  : ct = -1.0
        
        t[i] = np.arccos(ct)


def cartesian_to_spherical(dvec: ArrayF) -> Tuple[ArrayF64, ArrayF64, ArrayF64]:
    array = np.asarray(dvec, dtype=np.float64)
    if array.ndim == 1:
        if array.size != 3: raise ValueError("dvec must have 3 elements when 1D")
        array = array.reshape(1, 3)
    elif array.ndim == 2:
        if array.shape[1] != 3: raise ValueError("dvec must have shape (N, 3)")
    else: raise ValueError("dvec must be 1D or 2D")
    
    n = array.shape[0]
    x, y, z = array[:, 0], array[:, 1], array[:, 2]
    
    r = np.empty(n, dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)

    _cart2sph_kernel(x, y, z, r, p, t)
    
    # Convert to degrees and return
    return r, p * RAD2DEG, t * RAD2DEG



@njit(fastmath=True, parallel=True, cache=True)
def _sph2cart_kernel(
    r: np.ndarray, p: np.ndarray, t: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray
):
    """
    Numba kernel for spherical to Cartesian conversion. Takes the 
    angles measured in radians and compte the kernel in place.
    """
    n = len(r)
    for i in prange(n):
        ri, pi, ti = r[i], p[i], t[i]
        cp, ct, sp, st = np.cos(pi), np.cos(ti), np.sin(pi), np.sin(ti)

        x[i] = ri * cp * st
        y[i] = ri * sp * st
        z[i] = ri * ct


def spherical_to_cartesian(radius: ArrayF, phi: ArrayF, theta: ArrayF) -> ArrayF64:
    r, p, t = _as_1d_array(radius), _as_1d_array(phi), _as_1d_array(theta)

    n = max(len(r), len(p), len(t))
    if len(r) == 1: r = np.full(n, r[0])
    if len(p) == 1: p = np.full(n, p[0])
    if len(t) == 1: t = np.full(n, t[0])

    # Convert
    p, t = p * DEG2RAD, t * DEG2RAD

    # Pre-allocate output arrays
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    # use Numba kernel 
    _sph2cart_kernel(r, p, t, x, y, z)
    return np.column_stack((x, y, z))


# ----------======== Angle Combination (Rotation) ========---------- #

@njit(fastmath=True, cache=True)
def _rotation_matrix_elements(
    cp1: float64, sp1: float64, ct1: float64, st1: float64, inv: bool
) -> tuple:
    """
    Compute rotation matrix elements for given angles.. Returns 9 elements
    `m00`, `m01`, `m02`, `m10`, `m11`, `m12`, `m20`, `m21`, `m22`
    """
    if not inv: return (
        cp1 * ct1, -sp1, cp1 * st1,
        sp1 * ct1, cp1, sp1 * st1,
        -st1, 0.0, ct1
    )
    else: return (
        cp1 * ct1, sp1 * ct1, -st1,
        -sp1, cp1, 0.0,
        cp1 * st1, sp1 * st1, ct1
    )


@njit(fastmath=True, parallel=True, cache=True)
def _angle_rotation_kernel_numba(
    p0: np.ndarray, t0: np.ndarray, p1: np.ndarray, t1: np.ndarray, inv: bool = False
) -> tuple:
    """
    Numba - accelerated angle rotation kernel
    """
    n = len(p0)
    p_out, t_out = np.empty(n, np.float64), np.empty(n, np.float64)
    for i in prange(n):
        # Precompute pre-occurring trigonometric functions
        cp0, ct0, sp0, st0 = np.cos(p0[i]), np.cos(t0[i]), np.sin(p0[i]), np.sin(t0[i])
        cp1, ct1, sp1, st1 = np.cos(p1[i]), np.cos(t1[i]), np.sin(p1[i]), np.sin(t1[i])

        # original vectors 
        x0, y0, z0 = st0 * cp0, st0 * sp0, ct0

        # Get rotation matrix elements
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _rotation_matrix_elements(cp1, sp1, ct1, st1, inv)
        
        # Apply the rotation
        x = m00 * x0 + m01 * y0 + m02 * z0
        y = m10 * x0 + m11 * y0 + m12 * z0
        z = m20 * x0 + m21 * y0 + m22 * z0

        # Clamp z to avoid numerical issues
        z = min(max(z, -1.0), 1.0)

        # Convert back to spherical
        p_out[i], t_out[i] = np.arctan2(y, x), np.arccos(z)
        
    return p_out, t_out


# Numba accelerated version
def _angle_rotation_kernel(
    phi0: ArrayF, theta0: ArrayF, phi1: ArrayF, theta1: ArrayF, inv: bool = False
) -> Tuple[ArrayF64, ArrayF64]:
    """
    Rotate spherical angles using numba acceleration
    """
    #   Convert to numpy arrays
    p0, t0 = np.asarray(phi0, dtype=np.float64), np.asarray(theta0, dtype=np.float64)
    p1, t1 = np.asarray(phi1, dtype=np.float64), np.asarray(theta1, dtype=np.float64)

    # Broadcast if needed
    p0, t0, p1, t1 = np.broadcast_arrays(p0, t0, p1, t1)
    return _angle_rotation_kernel_numba(p0, t0, p1, t1, inv)


def _combine_angles(
    p0: ArrayF64, t0: ArrayF64, p1: ArrayF64, t1: ArrayF64, inv: bool=False
) -> Tuple[ArrayF64, ArrayF64]:
    p0, t0 = _as_1d_array(p0) * DEG2RAD, _as_1d_array(t0) * DEG2RAD
    p1, t1 = _as_1d_array(p1) * DEG2RAD, _as_1d_array(t1) * DEG2RAD

    # Broadcast to common shape
    p0, t0, p1, t1 = np.broadcast_arrays(p0, t0, p1, t1)

    # Apply rotation and the return back to degrees when return angles
    phi, theta = _angle_rotation_kernel_numba(p0, t0, p1, t1, inv)
    return phi * RAD2DEG, theta * RAD2DEG


def add_angles(
    phi0: ArrayF, theta0: ArrayF, phi1: ArrayF, theta1: ArrayF
) -> Tuple[ArrayF64, ArrayF64]:
    return _combine_angles(phi0, theta0, phi1, theta1, inv=False)


def sub_angles(
    phi0: ArrayF, theta0: ArrayF, phi1: ArrayF, theta1: ArrayF
) -> Tuple[ArrayF64, ArrayF64]:
    return _combine_angles(phi0, theta0, phi1, theta1, inv=True)
