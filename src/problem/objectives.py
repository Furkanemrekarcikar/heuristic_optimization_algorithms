"""
Objective function evaluation for the MODP.

Chosen objectives (3 required):
  f1 — User preference  → MAXIMIZE  (stored negated for unified minimization)
  f2 — Cost             → MINIMIZE
  f3 — CO₂ footprint    → MINIMIZE

All returned values are in MINIMIZATION form:
  objectives[0] = -sum(preferences)   (negate so lower is better)
  objectives[1] =  sum(costs)
  objectives[2] =  sum(co2s)
"""

from __future__ import annotations

import numpy as np

from ..database.loader import DataStore
from .chromosome import DecodedMenu


def compute_objectives(menu: DecodedMenu, ds: DataStore) -> np.ndarray:
    """
    Compute the 3 objective values from a decoded menu.

    Returns ndarray shape (3,) in minimization form:
        [−preference_sum, cost_sum, co2_sum]
    """
    selected = menu.breakfast_mask | menu.lunch_dinner_mask

    pref = ds.preferences[selected].sum()
    cost = ds.costs[selected].sum()
    co2  = ds.co2s[selected].sum()

    return np.array([-pref, cost, co2], dtype=np.float64)


def n_selected(menu: DecodedMenu) -> int:
    """Total number of food items in the menu."""
    return int((menu.breakfast_mask | menu.lunch_dinner_mask).sum())


def objective_names() -> list[str]:
    return ["Preference (negated)", "Cost", "CO2"]
