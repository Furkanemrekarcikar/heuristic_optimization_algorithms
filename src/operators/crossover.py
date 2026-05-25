"""
Order Crossover (OX) applied independently to each chromosome part.

Why OX over PMX:
  OX preserves relative ORDER of elements from one parent while respecting
  the positional contribution of the other — well-suited for the greedy
  decoder where position encodes priority.

Permutation invariant:
  offspring1_breakfast  = permutation of c1's breakfast IDs
  offspring1_lunch      = permutation of c1's lunch+dinner IDs
  → offspring1 is a valid permutation of 0..404. Same for offspring2/c2.

Filler projection:
  Each chromosome half may hold a DIFFERENT set of food IDs, so we cannot
  feed c2's breakfast array directly as the filler for c1's breakfast OX.
  Instead we project: extract c2's elements that belong to c1's half (in c2's
  original order). This guarantees donor and filler always share the same
  element set, which is the precondition for OX correctness.
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

    # Project each parent's full chromosome onto the OTHER parent's ID sets
    # so that donor and filler always contain exactly the same elements.
    b1 = _ox_part(c1[:s], c2[np.isin(c2, c1[:s])])   # child1 breakfast
    l1 = _ox_part(c1[s:], c2[np.isin(c2, c1[s:])])   # child1 lunch+dinner
    b2 = _ox_part(c2[:s], c1[np.isin(c1, c2[:s])])   # child2 breakfast
    l2 = _ox_part(c2[s:], c1[np.isin(c1, c2[s:])])   # child2 lunch+dinner

    return np.concatenate([b1, l1]), np.concatenate([b2, l2])


# ── Internal ──────────────────────────────────────────────────────────────────

def _ox_part(donor: np.ndarray, filler: np.ndarray) -> np.ndarray:
    """OX on a single part; donor and filler must share the same element set."""
    n = len(donor)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    return _ox_one(donor, filler, a, b)


def _ox_one(donor: np.ndarray, filler: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Single OX offspring:
      1. Copy donor[a:b] into the same positions.
      2. Fill remaining slots (left to right from b, wrapping)
         with filler elements not already placed.
    """
    n = len(donor)
    child      = np.empty(n, dtype=donor.dtype)
    child[a:b] = donor[a:b]
    placed     = set(donor[a:b].tolist())

    write = b
    for offset in range(n):
        val = filler[(b + offset) % n]
        if val not in placed:
            child[write % n] = val
            placed.add(val)
            write += 1

    return child
