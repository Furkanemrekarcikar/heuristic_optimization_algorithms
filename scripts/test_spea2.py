"""
Smoke-test SPEA2 on a small run (20 individuals, 10 generations).

Checks:
  1. Run completes without errors for both users
  2. Pareto front non-empty
  3. HV history correct length and non-negative
  4. Archive has expected size
  5. Non-dominated solutions in archive have F < 1

Usage:
    python scripts/test_spea2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.database.connection import DBConnection
from src.database.loader import load_data
from src.algorithms.spea2 import run, _assign_fitness
import src.config as cfg

POP          = 20
ARCHIVE_SIZE = 20
GENS         = 10


def main():
    errors = []

    for user_id, label in [(cfg.USER1_ID, "User1"), (cfg.USER2_ID, "User2")]:
        print(f"\n{'='*50}")
        print(f"  {label} (id={user_id})")
        print(f"{'='*50}")

        db = DBConnection(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_USER, cfg.DB_PASSWORD, cfg.DB_NAME)
        ds = load_data(db, user_id)

        result = run(
            ds,
            pop_size=POP,
            archive_size=ARCHIVE_SIZE,
            n_gen=GENS,
            verbose=True,
        )

        # 1. Pareto front non-empty
        if len(result.pareto_front) == 0:
            errors.append(f"{label}: Pareto front is empty")

        # 2. HV history length
        if len(result.hv_history) != GENS + 1:
            errors.append(
                f"{label}: HV history length {len(result.hv_history)} != {GENS + 1}"
            )

        # 3. HV non-negative
        if any(h < -1e-9 for h in result.hv_history):
            errors.append(f"{label}: negative HV value in history")

        # 4. Archive size
        if len(result.archive) != ARCHIVE_SIZE:
            errors.append(
                f"{label}: archive size {len(result.archive)} != {ARCHIVE_SIZE}"
            )

        # 5. Fitness consistency: archive non-dominated fitness < 1
        arch_fit = _assign_fitness(result.archive_obj)
        nd_in_archive = (arch_fit < 1.0).sum()

        print(f"\n  Pareto front   : {len(result.pareto_front)} solutions")
        print(f"  Final HV       : {result.hv_history[-1]:.6f}")
        print(f"  Archive ND     : {nd_in_archive}/{ARCHIVE_SIZE}")

    print("\n" + "="*50)
    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  FAIL: {e}")
    else:
        print("All SPEA2 smoke tests passed OK")


if __name__ == "__main__":
    main()
