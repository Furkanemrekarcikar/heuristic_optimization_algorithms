"""
Chromosome representation and greedy decoder for the MODP.

Representation
--------------
A chromosome is a permutation of 0..404 (array indices into DataStore arrays),
split at position BREAKFAST_SIZE into two independent parts:

    [ idx_0, idx_1, ..., idx_93  |  idx_94, ..., idx_404 ]
      <──── breakfast (94) ──────>  <──── lunch+dinner (311) ────>

The DB loads foods ORDER BY f.id; IDs 1..94 are breakfast foods (indices 0..93)
and IDs 95..405 are lunch+dinner foods (indices 94..404). Both parts are
shuffled independently so each part only ever contains its own food type.

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


# Food groups that are non-vegetarian and must be skipped for vegetarian users.
# Group 2 (Chicken Products) is excluded because it contains eggs.
_NON_VEG_GROUP_IDS: frozenset[int] = frozenset({
    3,   # Meat Products
    15,  # Chicken and Turkey based dishes
    23,  # Fish
    28,  # Meat based dishes
})


# ── Return type ───────────────────────────────────────────────────────────────

class DecodedMenu(NamedTuple):
    breakfast_mask:    np.ndarray  # bool (n_foods,) — selected breakfast items
    lunch_dinner_mask: np.ndarray  # bool (n_foods,) — selected lunch+dinner items
    nutrient_totals:   np.ndarray  # float (5,) — daily nutrient totals


# ── Public functions ──────────────────────────────────────────────────────────

def random_chromosome(n_foods: int = cfg.TOTAL_FOODS) -> np.ndarray:
    """
    Generate a random chromosome with two independently shuffled parts.

    Breakfast part  : a permutation of indices 0..BREAKFAST_SIZE-1
                      (DB food IDs 1..94 — actual breakfast foods)
    Lunch+dinner part: a permutation of indices BREAKFAST_SIZE..n_foods-1
                      (DB food IDs 95..405 — actual lunch+dinner foods)

    Keeping each part within its own index range ensures the decoder only
    evaluates breakfast foods during the breakfast phase and vice versa.
    """
    breakfast_part    = np.random.permutation(np.arange(0, cfg.BREAKFAST_SIZE))
    lunch_dinner_part = np.random.permutation(np.arange(cfg.BREAKFAST_SIZE, n_foods))
    return np.concatenate([breakfast_part, lunch_dinner_part])


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
        if _skip(ds, idx):
            continue
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
        if _skip(ds, idx):
            continue
        nuts = ds.nutrient_matrix[idx]
        if np.any(daily_totals + nuts > rul_d):
            continue
        ld_mask[idx] = True
        daily_totals += nuts
        if np.all(daily_totals >= rll_d):
            break

    return DecodedMenu(b_mask, ld_mask, daily_totals)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _skip(ds: DataStore, idx: int) -> bool:
    """Return True if this food should be skipped during decoding."""
    if ds.preferences[idx] < 0:
        return True
    # DB data is inconsistent for User 2: some non-veg foods have positive
    # preference. Group-based filter catches meat/fish that slipped through.
    if int(ds.food_group_ids[idx]) in _NON_VEG_GROUP_IDS:
        return ds.user_id == cfg.USER2_ID
    return False
