"""
Exact 3-D hypervolume indicator (minimization).

Algorithm: slice-by-slice.
  Sort points by obj3 ascending. Between consecutive z-events the 2-D
  cross-section is constant; it equals the 2-D HV of all points whose
  z-coordinate is <= the current slab's lower bound.

Reference point must strictly dominate all front points.
All objectives are in minimization form (lower = better).
"""

from __future__ import annotations

import numpy as np


def hypervolume(pareto_front: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the exact hypervolume of a 3-D Pareto front.

    Parameters
    ----------
    pareto_front : ndarray shape (n, 3)  — minimization objectives
    ref          : ndarray shape (3,)    — reference point (must dominate all)

    Returns
    -------
    float  hypervolume value (>= 0)
    """
    pts = np.asarray(pareto_front, dtype=float)
    ref = np.asarray(ref, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pareto_front must have shape (n, 3)")

    # Keep only points that are strictly dominated by the reference in all dims
    mask = np.all(pts < ref, axis=1)
    pts  = pts[mask]
    if len(pts) == 0:
        return 0.0

    # Sort ascending by the third objective
    pts = pts[np.argsort(pts[:, 2])]

    total = 0.0
    n     = len(pts)

    for i in range(n):
        # Slab: z in [pts[i,2], next_z)
        next_z = pts[i + 1, 2] if i < n - 1 else ref[2]
        dz     = next_z - pts[i, 2]
        if dz <= 0:
            continue
        # Active 2-D points: all with z <= pts[i,2]  (indices 0..i inclusive)
        total += _hv2d(pts[: i + 1, :2], ref[:2]) * dz

    return total


# ── Internal ──────────────────────────────────────────────────────────────────

def _hv2d(pts2d: np.ndarray, ref2: np.ndarray) -> float:
    """
    2-D hypervolume for a set of points (minimization).

    Sort by x ascending, sweep left-to-right tracking the running minimum y
    (staircase front). Each point that improves y contributes a rectangle
    of width (ref_x - x) and height (prev_y - y).
    """
    # Filter out points that do not lie strictly inside the reference box
    mask = (pts2d[:, 0] < ref2[0]) & (pts2d[:, 1] < ref2[1])
    pts  = pts2d[mask]
    if len(pts) == 0:
        return 0.0

    # Sort by x ascending; ties broken by y ascending (non-dom front stable)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    hv        = 0.0
    current_y = ref2[1]

    for x, y in pts:
        if y < current_y:
            hv        += (ref2[0] - x) * (current_y - y)
            current_y  = y

    return hv
