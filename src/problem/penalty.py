"""
Penalty function and diversity mechanism for the MODP.

DRI Penalty (Section 4 of handout)
------------------------------------
For each of the 5 nutrients j, with daily menu total vⱼ:

    viol_low_j  = max(0, RLL_j − vⱼ)  / (RUL_j − RLL_j)
    viol_high_j = max(0, vⱼ − RUL_j)  / (RUL_j − RLL_j)
    R           = 0.7·Σ viol_low_j  +  0.3·Σ viol_high_j

Under-nutrition is penalized more heavily than over-nutrition.
Raw (non-epsilon) DRI bounds are used here — the epsilon values only
widen the feasible space during greedy decoding, not for scoring.

Diversity Penalty (Section 5, Option B)
-----------------------------------------
    R_diversity = α · (1 / distinct_food_group_count)

Combined:
    R_total = R_dri + R_diversity   (α=0 → diversity disabled)

Applying to minimization objectives:
    penalized_objectives = objectives + λ · R_total

    λ starts at 1.0 (tune experimentally).
    A higher λ forces the algorithm toward feasibility at the cost of
    exploration; too low and infeasible solutions pollute the Pareto front.
"""

from __future__ import annotations

import numpy as np

from ..database.loader import DataStore
from .chromosome import DecodedMenu
from .. import config as cfg


def compute_penalty(
    menu: DecodedMenu,
    ds: DataStore,
    alpha: float = cfg.ALPHA_DIVERSITY,
) -> tuple[float, float, float]:
    """
    Compute DRI + diversity penalties.

    Parameters
    ----------
    menu  : DecodedMenu — output of decode()
    ds    : DataStore
    alpha : diversity penalty weight (0 = disabled)

    Returns
    -------
    (R_total, R_dri, R_diversity) — all non-negative floats
    """
    R_dri       = _dri_penalty(menu.nutrient_totals, ds)
    R_diversity = _diversity_penalty(menu, ds, alpha)
    return R_dri + R_diversity, R_dri, R_diversity


def apply_penalty(
    objectives: np.ndarray,
    R_total: float,
    lam: float = cfg.LAMBDA_PENALTY,
) -> np.ndarray:
    """
    Apply penalty to objectives stored in minimization form.

        penalized = objectives + λ · R_total

    Adding (not subtracting) because our objectives are already negated /
    oriented so that lower = better. Adding R makes worse solutions worse.
    """
    return objectives + lam * R_total


# ── Private helpers ───────────────────────────────────────────────────────────

def _dri_penalty(nutrient_totals: np.ndarray, ds: DataStore) -> float:
    """
    R_dri = 0.7·Σ viol_low_j + 0.3·Σ viol_high_j  for j=1..5
    Uses raw DRI bounds (no epsilon).
    """
    ranges = ds.dri_rul - ds.dri_rll  # (5,)  always > 0

    viol_low  = np.maximum(0.0, ds.dri_rll - nutrient_totals) / ranges
    viol_high = np.maximum(0.0, nutrient_totals - ds.dri_rul)  / ranges

    return float(0.7 * viol_low.sum() + 0.3 * viol_high.sum())


def _diversity_penalty(menu: DecodedMenu, ds: DataStore, alpha: float) -> float:
    """
    R_diversity = α · (1 / distinct_food_group_count)
    Target: at least 4–6 distinct food groups per daily menu.
    Returns 0 when alpha=0 (diversity disabled).
    """
    if alpha == 0.0:
        return 0.0

    selected = menu.breakfast_mask | menu.lunch_dinner_mask
    group_ids = ds.food_group_ids[selected]
    n_groups  = len(np.unique(group_ids)) if selected.any() else 0

    if n_groups == 0:
        return float(alpha)  # maximum penalty for empty menu
    return float(alpha / n_groups)
