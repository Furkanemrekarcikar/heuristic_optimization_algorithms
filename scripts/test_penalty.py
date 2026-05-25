"""
Test the penalty function and diversity mechanism.

Generates 10 random chromosomes for User 1 and shows:
  - DRI penalty, diversity penalty, total R
  - Penalized vs raw objectives
  - Distinct food group count per menu

Usage:
    python scripts/test_penalty.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.database.connection import DBConnection
from src.database.loader import load_data
from src.problem.chromosome import random_chromosome, decode
from src.problem.objectives import compute_objectives
from src.problem.penalty import compute_penalty, apply_penalty
import src.config as cfg

N = 10


def main():
    db = DBConnection(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_USER, cfg.DB_PASSWORD, cfg.DB_NAME)
    ds = load_data(db, cfg.USER1_ID)

    print(f"\nPenalty test — User 1 (λ={cfg.LAMBDA_PENALTY}, α={cfg.ALPHA_DIVERSITY})")
    print(f"{'#':>2}  {'R_dri':>7}  {'R_div':>7}  {'R_tot':>7}  "
          f"{'Groups':>6}  {'pref_raw':>9}  {'pref_pen':>9}  feasible")
    print("-" * 75)

    n_feasible = 0
    for i in range(N):
        chrom  = random_chromosome(ds.n_foods)
        menu   = decode(chrom, ds)
        objs   = compute_objectives(menu, ds)
        R_tot, R_dri, R_div = compute_penalty(menu, ds)
        pen_objs = apply_penalty(objs, R_tot)

        selected = menu.breakfast_mask | menu.lunch_dinner_mask
        n_groups = len(np.unique(ds.food_group_ids[selected])) if selected.any() else 0

        feasible = (R_dri == 0.0)
        if feasible:
            n_feasible += 1

        print(
            f"{i+1:>2}  {R_dri:>7.4f}  {R_div:>7.4f}  {R_tot:>7.4f}  "
            f"{n_groups:>6}  {-objs[0]:>9.2f}  {-pen_objs[0]:>9.2f}  "
            f"{'✓' if feasible else '✗'}"
        )

    print(f"\nFeasible (R_dri=0): {n_feasible}/{N}")
    print(f"Diversity range: {cfg.ALPHA_DIVERSITY:.2f}/n_groups  "
          f"(target ≥4 groups → R_div ≤ {cfg.ALPHA_DIVERSITY/4:.4f})")


if __name__ == "__main__":
    main()
