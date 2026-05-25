"""
Chromosome representation and greedy decoder for the MODP.

Representation
--------------
A chromosome is a permutation of 0..404 (array indices into DataStore arrays),
split at position BREAKFAST_SIZE into two independent parts:

    [ idx_0, idx_1, ..., idx_93  |  idx_94, ..., idx_404 ]
      <──── breakfast (94) ──────>  <──── lunch+dinner (311) ────>

The same food ID appears at most once across the full chromosome.

Decoding (greedy, left-to-right)
---------------------------------
Breakfast part — only Energy (col 0) and Protein (col 1) are checked:
  • Skip food if adding it would push Energy OR Protein above ε·RUL_b
  • Stop when Energy AND Protein have reached ε·RLL_b
  ε·RUL_b = RUL × BREAKFAST_RATIO × EPS_UPPER
  ε·RLL_b = RLL × BREAKFAST_RATIO × EPS_LOWER

Lunch+dinner part — all 5 nutrients checked against DAILY totals:
  • Carry over all 5 nutrient totals accumulated from breakfast
  • Skip food if adding it would push ANY nutrient above ε·RUL_d
  • Stop when ALL 5 nutrients have reached ε·RLL_d
  ε·RUL_d = RUL × EPS_UPPER
  ε·RLL_d = RLL × EPS_LOWER
"""

from __future__ import annotations
from typing import NamedTuple

import numpy as np

from ..database.loader import DataStore
from .. import config as cfg


# ── Return type ───────────────────────────────────────────────────────────────

class DecodedMenu(NamedTuple):
    breakfast_mask:    np.ndarray  # bool (n_foods,) — selected breakfast items
    lunch_dinner_mask: np.ndarray  # bool (n_foods,) — selected lunch+dinner items
    nutrient_totals:   np.ndarray  # float (5,) — daily nutrient totals


# ── Public functions ──────────────────────────────────────────────────────────

def random_chromosome(n_foods: int = cfg.TOTAL_FOODS) -> np.ndarray:
    """
    Generate a random chromosome: a permutation of 0..n_foods-1.
    The first BREAKFAST_SIZE elements form the breakfast part;
    the rest form the lunch+dinner part.
    Both parts are shuffled independently so the initial ordering within
    each part is unbiased.
    """
    perm = np.random.permutation(n_foods)
    # Shuffle each part independently (already random, but explicit for clarity)
    np.random.shuffle(perm[:cfg.BREAKFAST_SIZE])
    np.random.shuffle(perm[cfg.BREAKFAST_SIZE:])
    return perm


def decode(chromosome: np.ndarray, ds: DataStore) -> DecodedMenu:
    """
    Decode a chromosome permutation into a daily menu using the greedy procedure
    described in the handout (Section 3).

    Parameters
    ----------
    chromosome : ndarray shape (n_foods,)
        Permutation of 0..n_foods-1 indices.
    ds : DataStore
        Pre-loaded data for a specific user.

    Returns
    -------
    DecodedMenu with breakfast_mask, lunch_dinner_mask, nutrient_totals.
    """
    # ── Pre-compute effective DRI bounds ──────────────────────────────────────
    # Breakfast: Energy (col 0) + Protein (col 1) only, 35% of daily DRI
    rul_b = ds.dri_rul[:2] * cfg.BREAKFAST_RATIO * cfg.EPS_UPPER  # (2,)
    rll_b = ds.dri_rll[:2] * cfg.BREAKFAST_RATIO * cfg.EPS_LOWER  # (2,)

    # Daily totals: all 5 nutrients
    rul_d = ds.dri_rul * cfg.EPS_UPPER  # (5,)
    rll_d = ds.dri_rll * cfg.EPS_LOWER  # (5,)

    n = ds.n_foods
    b_genes  = chromosome[:cfg.BREAKFAST_SIZE]
    ld_genes = chromosome[cfg.BREAKFAST_SIZE:]

    # ── Breakfast decoding ────────────────────────────────────────────────────
    b_mask = np.zeros(n, dtype=bool)
    b_ep   = np.zeros(2, dtype=np.float64)  # running Energy + Protein totals

    for idx in b_genes:
        ep = ds.nutrient_matrix[idx, :2]
        if np.any(b_ep + ep > rul_b):
            continue
        b_mask[idx] = True
        b_ep += ep
        if np.all(b_ep >= rll_b):
            break

    # Accumulate ALL 5 nutrient totals from breakfast (not just E+P)
    b_selected = np.where(b_mask)[0]
    daily_totals = (
        ds.nutrient_matrix[b_selected].sum(axis=0)
        if len(b_selected) > 0
        else np.zeros(5, dtype=np.float64)
    )

    # ── Lunch+dinner decoding ─────────────────────────────────────────────────
    ld_mask = np.zeros(n, dtype=bool)

    for idx in ld_genes:
        nuts = ds.nutrient_matrix[idx]
        if np.any(daily_totals + nuts > rul_d):
            continue
        ld_mask[idx] = True
        daily_totals += nuts
        if np.all(daily_totals >= rll_d):
            break

    return DecodedMenu(b_mask, ld_mask, daily_totals)
