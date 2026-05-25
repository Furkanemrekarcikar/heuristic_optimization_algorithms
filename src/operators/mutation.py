"""
Swap mutation applied independently to each chromosome part.

p_m = 1/n  (n = part length) means on average one swap per part per generation.
Each part is mutated independently: a random pair within the part is swapped.
"""

from __future__ import annotations

import numpy as np

from .. import config as cfg


def mutate(
    chromosome: np.ndarray,
    p_m_b:  float | None = None,
    p_m_ld: float | None = None,
) -> np.ndarray:
    """
    Apply swap mutation to breakfast and lunch+dinner parts independently.

    Parameters
    ----------
    chromosome : ndarray shape (n_foods,)
    p_m_b      : mutation probability for breakfast part  (default 1/94)
    p_m_ld     : mutation probability for lunch+dinner part (default 1/311)

    Returns a mutated copy (original is not modified).
    """
    s   = cfg.BREAKFAST_SIZE
    n_b = s
    n_l = len(chromosome) - s

    result = chromosome.copy()
    _swap_part(result, 0, s,           p_m_b  if p_m_b  is not None else 1.0 / n_b)
    _swap_part(result, s, len(result), p_m_ld if p_m_ld is not None else 1.0 / n_l)
    return result


# ── Internal ──────────────────────────────────────────────────────────────────

def _swap_part(arr: np.ndarray, start: int, end: int, p_m: float) -> None:
    """In-place single swap on arr[start:end] with probability p_m."""
    if np.random.random() < p_m:
        n = end - start
        i, j = np.random.choice(n, 2, replace=False)
        arr[start + i], arr[start + j] = arr[start + j], arr[start + i]
