"""
Binary tournament selection.

The comparison logic is injected via `is_better(i, j) -> bool` so this
function works for both NSGA-II (rank + crowding distance) and SPEA2
(scalar strength+density fitness).

NSGA-II comparison:
    is_better = lambda i, j: (rank[i] < rank[j]) or
                             (rank[i] == rank[j] and crowd[i] > crowd[j])

SPEA2 comparison:
    is_better = lambda i, j: fitness[i] < fitness[j]
"""

from __future__ import annotations
from typing import Callable

import numpy as np


def binary_tournament(
    n_pop: int,
    n_select: int,
    is_better: Callable[[int, int], bool],
) -> np.ndarray:
    """
    Select `n_select` individuals from a population of size `n_pop` using
    binary tournament selection.

    Parameters
    ----------
    n_pop     : population size
    n_select  : number of individuals to select
    is_better : callable(i, j) → True if individual i beats individual j

    Returns
    -------
    ndarray shape (n_select,) of selected population indices
    """
    candidates = np.random.randint(0, n_pop, size=(n_select, 2))
    selected   = np.where(
        [is_better(a, b) for a, b in candidates],
        candidates[:, 0],
        candidates[:, 1],
    )
    return selected
