"""
Verify database connection and data loading.
Prints a summary of what was loaded for User 1 and User 2.

Usage:
    python scripts/test_db.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.database.connection import DBConnection
from src.database.loader import load_data
import src.config as cfg


def summarize(label: str, ds) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}  (user_id={ds.user_id})")
    print(f"{'='*50}")
    print(f"  Foods loaded      : {ds.n_foods}")
    print(f"  Food groups       : {len(ds.food_groups)}")

    print(f"\n  DRI bounds (user profile):")
    for j, nid in enumerate(cfg.CONSTRAINT_NUTRIENT_IDS):
        name = cfg.NUTRIENT_NAMES[nid]
        print(f"    {name:<22}  RLL={ds.dri_rll[j]:>8.2f}  RUL={ds.dri_rul[j]:>8.2f}")

    print(f"\n  Objective ranges:")
    print(f"    Preference : [{ds.preferences.min():.2f}, {ds.preferences.max():.2f}]  "
          f"  negative={np.sum(ds.preferences < 0)}")
    print(f"    Cost       : [{ds.costs.min():.2f}, {ds.costs.max():.2f}]")
    print(f"    Prep time  : [{ds.prep_times.min():.1f}, {ds.prep_times.max():.1f}] min")
    print(f"    CO2        : [{ds.co2s.min():.2f}, {ds.co2s.max():.2f}]")

    print(f"\n  Nutrient matrix  : shape {ds.nutrient_matrix.shape}")
    zero_rows = np.sum(np.all(ds.nutrient_matrix == 0, axis=1))
    print(f"    Foods with ALL zero nutrients : {zero_rows}")

    print(f"\n  Sample foods (first 5):")
    for i in range(min(5, ds.n_foods)):
        print(f"    [{ds.food_ids[i]:>3}] {ds.food_names[i]:<35} "
              f"pref={ds.preferences[i]:>5.1f}  cost={ds.costs[i]:>5.2f}")


def main():
    db = DBConnection(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_USER, cfg.DB_PASSWORD, cfg.DB_NAME)

    print("Testing connection...", end=" ")
    if not db.test():
        print("FAILED — check DB credentials in src/config.py")
        sys.exit(1)
    print("OK")

    for uid, label in [(cfg.USER1_ID, "User 1 — Non-vegetarian"),
                       (cfg.USER2_ID, "User 2 — Vegetarian")]:
        try:
            ds = load_data(db, uid)
            summarize(label, ds)
        except Exception as exc:
            print(f"\nERROR loading {label} (user_id={uid}): {exc}")
            print("Run `python scripts/list_users.py` to find valid user IDs.")

    print()


if __name__ == "__main__":
    main()
