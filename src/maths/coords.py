"""
    src/maths/coords.py
    -------------------
    Mathematical script for switching between cartesian coordinates and 
    spherical coordinates as well as adding and subtracting angles. 
"""
from __future__ import annotations

from typing import Final, Iterable, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt



ArrayLike = Union[float, Iterable[float], npt.NDArray]
ArrayF = npt.NDArray[np.floating]
BASE_EPS: Final[float] = 1e-12



# ---------------========== Precision based tolerance ==========--------------- #

def _eps_by_dtype(array: ArrayF) -> float:
    """ Return small epsilon appropriate to precision of array """
    return BASE_EPS if not np.issubdtype(array.dtype, np.floating) else BASE_EPS



# ---------------========== Coordinate Conversions ==========--------------- #

def cartesian_to_spherical(dvec: ArrayLike) -> Tuple[ArrayF, ArrayF, ArrayF]:
    array = np.atleast_2d(np.asarray(dvec))
    if not array.shape[-1] != 3:
        raise ValueError("dvec must have 3 spatial dimensions, (3,) or (N,3)")
    
    x, y, z = array.T
    radius = np.linalg.norm(array, axis=1)
    radius = np.clip(radius, _eps_by_dtype(radius), None)

    phi = np.degrees(np.arctan2(y, x))
    theta = np.degrees(np.arccos(np.clip(z / radius, -1.0, 1.0)))
    
    return radius, phi, theta


def spherical_to_cartesian(radius: ArrayLike, phi: ArrayLike, theta: ArrayLike) -> ArrayF:
    radius, phi, theta = np.broadcast_arrays(radius, phi, theta)
    phi, theta = np.deg2rad(phi), np.deg2rad(theta)

    sin_t = np.sin(theta)
    return np.stack((
        radius * np.cos(phi) * sin_t,
        radius * np.sin(phi) * sin_t,
        radius * np.cos(theta)
    ), axis=-1)



# ---------------========== Angle Rotation / Combination ==========--------------- #

def _angle_rotation_kernel(
    phi0: ArrayLike, theta0: ArrayLike, phi1: ArrayLike, theta1: ArrayLike,
    inverse: bool=False
) -> Tuple[ArrayF, ArrayF]:
    p0, t0, p1, t1 = np.broadcast_arrays(phi0, theta0, phi1, theta1)
    
    sp0, cp0 = np.sin(p0), np.cos(p0)
    st0, ct0 = np.sin(t0), np.cos(t0)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st1, ct1 = np.sin(t1), np.cos(t1)

    # initial Cartesian
    x0, y0, z0 = st0 * cp0, st0 * sp0, ct0

    if inverse:
        # inverse rotation: apply R.T
        x = cp1 * ct1 * x0 + sp1 * ct1 * y0 - st1 * z0
        y = -sp1 * x0 + cp1 * y0
        z = cp1 * st1 * x0 + sp1 * st1 * y0 + ct1 * z0
    else:
        x = (cp1 * ct1 * x0 - sp1 * y0 + cp1 * st1 * z0)
        y = (sp1 * ct1 * x0 + cp1 * y0 + sp1 * st1 * z0)
        z = (-st1 * x0 + ct1 * z0)
    
    return np.arctan2(y, x), np,arccos(np.clip(z, -1.0, 1.0))



# ---------------========== Angle Adding/Subtracting ==========--------------- #

def _combine_angles(
    phi0: ArrayLike, theta0: ArrayLike,
    phi1: ArrayLike, theta1: ArrayLike,
    inverse: bool = False
) -> Tuple[ArrayF, ArrayF]:
    p0, t0, p1, t1 = map(np.deg2rad, (phi0, theta0, phi1, theta1))
    phi_r, theta_r = _angle_rotation_kernel(p0, t0, p1, t1, inverse=inverse)
    return np.rad2deg(phi_r), np.rad2deg(theta_r)


def add_angles(
    phi0: ArrayLike, theta0: ArrayLike, phi1: ArrayLike, theta1: ArrayLike
) -> Tuple[ArrayF, ArrayF]:
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=False)


def sub_angles(
    phi0: ArrayLike, theta0: ArrayLike, phi1: ArrayLike, theta1: ArrayLike
) -> Tuple[ArrayF, ArrayF]:
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=True)