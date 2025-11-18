"""
    maths/coords.py
    ---------------
"""
from __future__ import annotations

from typing import Final, Iterable, Tuple, Union
import numpy as np
import numpy.typing as npt

ArrayLike = Union[float, Iterable[float], npt.NDArray]
ArrayF = npt.NDArray[np.floating]
BASE_EPS: Final[float] = 1e-12


# ---------------========== Utilities ==========---------------- #

def _eps_for_dtype(array: ArrayF) -> float:
    """
    Return epsilon `base-tolerance` appropriate for the 
    dtype for the passed array.
    """
    if not np.issubdtype(array.dtype, np.floating):
        return BASE_EPS
    return float(np.finfo(array.dtype).eps * 50.0) 


def _normalize_angle_degree(phi: ArrayF) -> ArrayF:
    """
    Normalize the angle into the range [-180, 180]
    """
    return (phi + 180.0) % 360.0 - 180.0


# ---------------========== Cartesian <=> Spherical ==========---------------- #

def cartesian_to_spherical(dvec: ArrayLike) -> Tuple[ArrayF, ArrayF, ArrayF]:
    """
    Takes a vector/array and takes it cartesian coordinates `x`, `y` and `z`
    and perform the conversion to the spherical correspondence. The return 
    values are the radius `distance` of the vector, the azimuth as well as
    the elevation angles.
    """
    array = np.atleast_2d(np.asarray(dvec))
    if array.shape[-1] != 3:
        raise ValueError("dvec must have shape (3,) or (N,3)")

    x, y, z = array.T
    radius = np.linalg.norm(array, axis=1)
    radius = np.clip(radius, _eps_for_dtype(radius), None)

    phi = np.degrees(np.arctan2(y, x))
    theta = np.degrees(np.arccos(np.clip(z / radius, -1.0, 1.0)))
    return radius, phi, theta


def spherical_to_cartesian(
    radius: ArrayLike, phi: ArrayLike, theta: ArrayLike
) -> ArrayF:
    """
    Takes the radius, as well as the azimuth and elevation angles of 
    a vector representation. This method takes these argument information
    and convert it back into the corresponding `x`, `y`, and `z` coordinates.
    """
    r, p, t = np.broadcast_arrays(radius, phi, theta)
    p, t = np.deg2rad(p), np.deg2rad(t)

    sin_t = np.sin(t)
    x, y, z = r * np.cos(p) * sin_t, r * np.sin(p) * sin_t, r * np.cos(t)
    return np.stack((x, y, z), axis=-1)


# ---------------========== Angle Rotation / Composition ==========---------------- #

def _angle_rotation_kernel(
    p0: ArrayF, t0: ArrayF, p1: ArrayF, t1: ArrayF, inverse: bool
) -> Tuple[ArrayF, ArrayF]:
    """
    """
    sp0, cp0 = np.sin(p0), np.cos(p0)
    st0, ct0 = np.sin(t0), np.cos(t0)
    sp1, cp1 = np.sin(p1), np.cos(p1)
    st1, ct1 = np.sin(t1), np.cos(t1)

    # convert input angles to Cartesian
    x0, y0, z0 = st0 * cp0, st0 * sp0, ct0

    if inverse:
        # rotation transpose
        x = cp1 * ct1 * x0 + sp1 * ct1 * y0 - st1 * z0
        y = -sp1 * x0 + cp1 * y0
        z = cp1 * st1 * x0 + sp1 * st1 * y0 + ct1 * z0
    else:
        x = cp1 * ct1 * x0 - sp1 * y0 + cp1 * st1 * z0
        y = sp1 * ct1 * x0 + cp1 * y0 + sp1 * st1 * z0
        z = -st1 * x0 + ct1 * z0

    z = np.clip(z, -1.0, 1.0)
    return np.arctan2(y, x), np.arccos(z)


# def _angle_rotation_kernel(
#     phi0: ArrayLike, theta0: ArrayLike,
#     phi1: ArrayLike, theta1: ArrayLike,
#     inverse: bool = False
# ) -> Tuple[ArrayF, ArrayF]:
#     p0, t0, p1, t1 = np.broadcast_arrays(phi0, theta0, phi1, theta1)
#     sp0, cp0 = np.sin(p0), np.cos(p0)
#     st0, ct0 = np.sin(t0), np.cos(t0)
#     sp1, cp1 = np.sin(p1), np.cos(p1)
#     st1, ct1 = np.sin(t1), np.cos(t1)

#     # initial Cartesian
#     x0, y0, z0 = st0 * cp0, st0 * sp0, ct0

#     if inverse:
#         # inverse rotation: apply R.T
#         x = cp1 * ct1 * x0 + sp1 * ct1 * y0 - st1 * z0
#         y = -sp1 * x0 + cp1 * y0
#         z = cp1 * st1 * x0 + sp1 * st1 * y0 + ct1 * z0
#     else:
#         x = (cp1 * ct1 * x0 - sp1 * y0 + cp1 * st1 * z0)
#         y = (sp1 * ct1 * x0 + cp1 * y0 + sp1 * st1 * z0)
#         z = (-st1 * x0 + ct1 * z0)

#     z = np.clip(z, -1.0, 1.0)
#     return np.arctan2(y, x), np.arccos(z)


def _combine_angles(
    phi0: ArrayLike, theta0: ArrayLike,
    phi1: ArrayLike, theta1: ArrayLike, inverse: bool
) -> Tuple[ArrayF, ArrayF]:
    """
    """
    # Convert inputs to radians only once (faster)
    p0, t0, p1, t1 = map(np.radians, (phi0, theta0, phi1, theta1))

    phi, theta = _angle_rotation_kernel(p0, t0, p1, t1, inverse)
    return _normalize_angle_degree(np.degrees(phi)), np.degrees(theta)

# def _combine_angles(
#     phi0: ArrayLike, theta0: ArrayLike,
#     phi1: ArrayLike, theta1: ArrayLike,
#     inverse: bool = False
# ) -> Tuple[ArrayF, ArrayF]:
#     p0, t0, p1, t1 = map(np.deg2rad, (phi0, theta0, phi1, theta1))
#     phi_r, theta_r = _angle_rotation_kernel(p0, t0, p1, t1, inverse=inverse)
#     return np.rad2deg(phi_r), np.rad2deg(theta_r)


def add_angles(
    phi0: ArrayLike, theta0: ArrayLike,
    phi1: ArrayLike, theta1: ArrayLike
) -> Tuple[ArrayF, ArrayF]:
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=False)


def sub_angles(
    phi0: ArrayLike, theta0: ArrayLike,
    phi1: ArrayLike, theta1: ArrayLike
) -> Tuple[ArrayF, ArrayF]:
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=True)
