"""
Smoke-test for chromosome decoding and objective evaluation.

Decodes 5 random chromosomes for User 1 and prints:
  - Number of breakfast / lunch+dinner items selected
  - Nutrient totals vs DRI bounds (compliance)
  - Objective values

Usage:
    python scripts/test_chromosome.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.database.connection import DBConnection
from src.database.loader import load_data
from src.problem.chromosome import random_chromosome, decode
from src.problem.objectives import compute_objectives
import src.config as cfg


NUTRIENT_LABELS = ["Energy(kcal)", "Protein(g)", "Carb(g)", "Fiber(g)", "Sodium(mg)"]


def compliance_str(totals, rll, rul):
    parts = []
    for j in range(5):
        lo = rll[j] * cfg.EPS_LOWER
        hi = rul[j] * cfg.EPS_UPPER
        ok = lo <= totals[j] <= hi
        parts.append("✓" if ok else "✗")
    return " ".join(parts)


def print_menu_summary(idx, menu, ds, objs):
    n_b  = menu.breakfast_mask.sum()
    n_ld = menu.lunch_dinner_mask.sum()
    totals = menu.nutrient_totals

    print(f"\n  Chromosome #{idx+1}")
    print(f"    Items: {n_b} breakfast + {n_ld} lunch/dinner = {n_b+n_ld} total")
    print(f"    Nutrients vs DRI (effective bounds):")
    for j, name in enumerate(NUTRIENT_LABELS):
        lo = ds.dri_rll[j] * cfg.EPS_LOWER
        hi = ds.dri_rul[j] * cfg.EPS_UPPER
        ok = "✓" if lo <= totals[j] <= hi else "✗"
        print(f"      {ok} {name:<16} {totals[j]:>8.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"    Objectives:  preference={-objs[0]:.2f}  cost={objs[1]:.2f}  co2={objs[2]:.2f}")


def main():
    db = DBConnection(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_USER, cfg.DB_PASSWORD, cfg.DB_NAME)

    for user_id, label in [(cfg.USER1_ID, "User 1 — Non-vegetarian"),
                           (cfg.USER2_ID, "User 2 — Vegetarian")]:
        print(f"\n{'='*55}")
        print(f"  {label}  (user_id={user_id})")
        print(f"{'='*55}")

        ds = load_data(db, user_id)

        fully_feasible = 0
        for i in range(5):
            chrom = random_chromosome(ds.n_foods)
            menu  = decode(chrom, ds)
            objs  = compute_objectives(menu, ds)
            print_menu_summary(i, menu, ds, objs)

            totals = menu.nutrient_totals
            rll_eff = ds.dri_rll * cfg.EPS_LOWER
            rul_eff = ds.dri_rul * cfg.EPS_UPPER
            if np.all(totals >= rll_eff) and np.all(totals <= rul_eff):
                fully_feasible += 1

        print(f"\n  Fully feasible (all 5 nutrients within bounds): {fully_feasible}/5")

    print()


if __name__ == "__main__":
    main()
