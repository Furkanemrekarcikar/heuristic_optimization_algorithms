"""
Order Crossover (OX) applied independently to each chromosome part.

Why OX over PMX:
  OX preserves relative ORDER of elements from one parent while respecting
  the positional contribution of the other — well-suited for the greedy
  decoder where position encodes priority.

Permutation invariant:
  offspring_breakfast  = permutation of parent1's breakfast IDs
  offspring_lunch      = permutation of parent1's lunch+dinner IDs
  → offspring is a valid permutation of 0..404 as long as parent1 is.

Two offspring are produced per call (parent roles are swapped for child 2).
"""

from __future__ import annotations

import numpy as np

from .. import config as cfg


def crossover(
    c1: np.ndarray,
    c2: np.ndarray,
    p_c: float = cfg.P_CROSS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply OX crossover to the breakfast and lunch+dinner parts independently.

    With probability (1 - p_c) returns copies of the parents unchanged.
    """
    if np.random.random() >= p_c:
        return c1.copy(), c2.copy()

    s = cfg.BREAKFAST_SIZE

    b1, b2 = _ox(c1[:s],  c2[:s])
    l1, l2 = _ox(c1[s:],  c2[s:])

    return np.concatenate([b1, l1]), np.concatenate([b2, l2])


# ── Internal ──────────────────────────────────────────────────────────────────

def _ox(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard OX on two permutation arrays of the same length.
    Returns two offspring.
    """
    n = len(p1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    return _ox_one(p1, p2, a, b), _ox_one(p2, p1, a, b)


def _ox_one(donor: np.ndarray, filler: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Single OX offspring:
      1. Copy donor[a:b] into the same positions.
      2. Fill remaining slots (left to right from position b, wrapping)
         with elements from filler in order, skipping already-placed ones.
    """
    n = len(donor)
    child    = np.empty(n, dtype=donor.dtype)
    child[a:b] = donor[a:b]
    placed   = set(donor[a:b].tolist())

    write = b % n
    for offset in range(n):
        val = filler[(b + offset) % n]
        if val not in placed:
            child[write] = val
            placed.add(val)
            write = (write + 1) % n

    return child
