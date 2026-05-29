"""
Verify genetic operators preserve the permutation invariant and behave correctly.

Checks:
  1. OX crossover produces valid permutations (no duplicates, all 0..404 present)
  2. Breakfast / lunch+dinner ID sets are preserved across crossover
  3. Swap mutation produces valid permutations
  4. Crossover probability p_c is respected

Usage:
    python scripts/test_operators.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.problem.chromosome import random_chromosome
from src.operators.crossover import crossover
from src.operators.mutation import mutate
from src.operators.selection import binary_tournament
import src.config as cfg

N_TRIALS = 500


def is_valid_perm(arr, n):
    return len(arr) == n and len(set(arr.tolist())) == n and arr.min() == 0 and arr.max() == n - 1


def main():
    n = cfg.TOTAL_FOODS
    s = cfg.BREAKFAST_SIZE
    errors = []

    # ── 1. Crossover validity ─────────────────────────────────────────────────
    cross_count = 0
    for _ in range(N_TRIALS):
        p1 = random_chromosome(n)
        p2 = random_chromosome(n)
        c1, c2 = crossover(p1, p2, p_c=1.0)  # force crossover
        cross_count += 1

        for child, label in [(c1, "c1"), (c2, "c2")]:
            if not is_valid_perm(child, n):
                errors.append(f"Crossover: {label} is not a valid permutation")

        # Breakfast IDs of child must equal breakfast IDs of one parent
        p1_b = set(p1[:s].tolist())
        p2_b = set(p2[:s].tolist())
        c1_b = set(c1[:s].tolist())
        c2_b = set(c2[:s].tolist())
        if c1_b != p1_b:
            errors.append(f"Crossover: c1 breakfast IDs ≠ p1 breakfast IDs")
        if c2_b != p2_b:
            errors.append(f"Crossover: c2 breakfast IDs ≠ p2 breakfast IDs")

    # ── 2. Crossover probability ──────────────────────────────────────────────
    unchanged = sum(
        1 for _ in range(N_TRIALS)
        if np.array_equal(crossover(random_chromosome(n), random_chromosome(n), p_c=cfg.P_CROSS)[0],
                          random_chromosome(n))  # rough check — just count
    )
    # With p_c=0, crossover should always return copies
    p1 = random_chromosome(n)
    p2 = random_chromosome(n)
    c1, c2 = crossover(p1, p2, p_c=0.0)
    if not np.array_equal(c1, p1) or not np.array_equal(c2, p2):
        errors.append("Crossover: p_c=0.0 should return unchanged parents")

    # ── 3. Mutation validity ─────────────────────────────────────────────────
    for _ in range(N_TRIALS):
        p = random_chromosome(n)
        m = mutate(p)
        if not is_valid_perm(m, n):
            errors.append("Mutation: result is not a valid permutation")
        # Mutation should preserve breakfast ID set (swap is within each part)
        if set(m[:s].tolist()) != set(p[:s].tolist()):
            errors.append("Mutation: breakfast ID set changed after mutation")
        if set(m[s:].tolist()) != set(p[s:].tolist()):
            errors.append("Mutation: lunch+dinner ID set changed after mutation")

    # ── 4. Binary tournament ──────────────────────────────────────────────────
    fitness = np.random.rand(50)
    selected = binary_tournament(50, 20, is_better=lambda i, j: fitness[i] < fitness[j])
    if len(selected) != 20 or selected.min() < 0 or selected.max() >= 50:
        errors.append("Selection: output shape or range incorrect")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nOperator tests ({N_TRIALS} trials each)")
    print(f"  Crossover : {'OK' if not any('Crossover' in e for e in errors) else 'FAIL'}")
    print(f"  Mutation  : {'OK' if not any('Mutation'  in e for e in errors) else 'FAIL'}")
    print(f"  Selection : {'OK' if not any('Selection' in e for e in errors) else 'FAIL'}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in set(errors):
            print(f"  -- {e}")
    else:
        print("\nAll checks passed OK")


if __name__ == "__main__":
    main()
