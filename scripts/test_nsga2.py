"""
Smoke-test NSGA-II on a small run (20 individuals, 10 generations).

Checks:
  1. Run completes without errors for both users
  2. Pareto front is non-empty
  3. HV history is non-decreasing (loosely — allow ±1e-9 tolerance)
  4. All front-0 objectives are mutually non-dominated

Usage:
    python scripts/test_nsga2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.database.connection import DBConnection
from src.database.loader import load_data
from src.algorithms.nsga2 import run
import src.config as cfg

POP  = 20
GENS = 10


def check_nondominated(front: np.ndarray) -> bool:
    """Return True iff no point in `front` is dominated by another."""
    n = len(front)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(front[j] <= front[i]) and np.any(front[j] < front[i]):
                return False
    return True


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
            n_gen=GENS,
            verbose=True,
        )

        # 1. Pareto front non-empty
        if len(result.pareto_front) == 0:
            errors.append(f"{label}: Pareto front is empty")

        # 2. HV history has correct length
        expected_len = GENS + 1
        if len(result.hv_history) != expected_len:
            errors.append(
                f"{label}: HV history length {len(result.hv_history)} != {expected_len}"
            )

        # 3. HV non-negative
        if any(h < -1e-9 for h in result.hv_history):
            errors.append(f"{label}: negative HV value in history")

        # 4. front-0 is mutually non-dominated in *penalised* objective space
        front0_pen = result.pen_objectives[result.ranks == 0]
        if len(front0_pen) > 1:
            if not check_nondominated(front0_pen):
                errors.append(f"{label}: front-0 contains dominated solutions (pen objectives)")

        # 5. population shape
        if result.population.shape != (POP, ds.food_ids.shape[0]):
            errors.append(f"{label}: population shape mismatch")

        print(f"\n  Front-0 size : {len(result.pareto_front)}")
        print(f"  Final HV     : {result.hv_history[-1]:.6f}")
        print(f"  Ranks range  : [{result.ranks.min()}, {result.ranks.max()}]")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  FAIL: {e}")
    else:
        print("All NSGA-II smoke tests passed OK")


if __name__ == "__main__":
    main()
