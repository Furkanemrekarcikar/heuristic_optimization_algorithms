"""
SPEA2 (Strength Pareto Evolutionary Algorithm 2) for the MODP.

Zitzler, Laumanns & Thiele (2001):
  F(i) = R(i) + D(i)

  S(i) = number of solutions dominated by i  (strength)
  R(i) = sum of S(j) for all j that dominate i  (raw fitness)
  D(i) = 1 / (sigma_k + 2)  where sigma_k is the k-NN distance in objective
         space, k = floor(sqrt(|P| + |A|))

  F(i) < 1  →  non-dominated (survives to archive unconditionally)

Archive selection:
  1. Keep all F < 1.
  2. Too few → fill with best dominated (ascending F).
  3. Too many → prune iteratively: remove the individual whose nearest
     neighbour is closest; break ties by next-nearest, etc.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..database.loader import DataStore
from ..problem.chromosome import random_chromosome, decode
from ..problem.objectives import compute_objectives
from ..problem.penalty import compute_penalty, apply_penalty
from ..operators.crossover import crossover
from ..operators.mutation import mutate
from ..operators.selection import binary_tournament
from ..metrics.hypervolume import hypervolume
from .. import config as cfg


# ── Public result type ────────────────────────────────────────────────────────

class SPEA2Result(NamedTuple):
    population:     np.ndarray   # shape (pop_size, n_foods) — final population
    archive:        np.ndarray   # shape (archive_size, n_foods)
    objectives:     np.ndarray   # shape (pop_size, 3) — raw objectives
    archive_obj:    np.ndarray   # shape (archive_size, 3)
    fitness:        np.ndarray   # shape (pop_size + archive_size,) — SPEA2 F values
    pareto_front:   np.ndarray   # shape (n_nd, 3) — non-dominated raw objectives
    hv_history:     list[float]


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    ds: DataStore,
    pop_size: int         = cfg.POP_SIZE,
    archive_size: int     = cfg.POP_SIZE,
    n_gen: int            = cfg.N_GEN,
    p_c: float            = cfg.P_CROSS,
    lam: float            = cfg.LAMBDA_PENALTY,
    alpha: float          = cfg.ALPHA_DIVERSITY,
    hv_ref: np.ndarray | None = None,
    verbose: bool         = True,
) -> SPEA2Result:
    """
    Run SPEA2.

    Parameters
    ----------
    ds           : DataStore loaded for the target user
    pop_size     : population size
    archive_size : external archive capacity
    n_gen        : number of generations
    p_c          : crossover probability
    lam          : penalty weight λ
    alpha        : diversity penalty weight α
    hv_ref       : HV reference point (shape (3,)). Auto-set if None.
    verbose      : print progress every 10 generations
    """
    n = ds.food_ids.shape[0]

    # ── Initialise ────────────────────────────────────────────────────────────
    pop     = np.array([random_chromosome(n) for _ in range(pop_size)])
    raw_pop, pen_pop = _evaluate(pop, ds, lam, alpha)

    # Empty archive at start
    archive     = np.empty((0, n), dtype=pop.dtype)
    raw_archive = np.empty((0, 3))
    pen_archive = np.empty((0, 3))

    if hv_ref is None:
        hv_ref = raw_pop.max(axis=0) * 1.1 + 1e-6

    hv_history: list[float] = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for gen in range(n_gen):
        # Combine population + archive
        combined     = np.concatenate([pop, archive], axis=0)
        raw_combined = np.concatenate([raw_pop, raw_archive], axis=0)
        pen_combined = np.concatenate([pen_pop, pen_archive], axis=0)

        # SPEA2 fitness assignment
        fitness = _assign_fitness(pen_combined)

        # Track HV of current non-dominated set
        nd_mask = fitness < 1.0
        hv_val  = hypervolume(raw_combined[nd_mask], hv_ref) if nd_mask.any() else 0.0
        hv_history.append(hv_val)

        if verbose and gen % 10 == 0:
            print(f"  Gen {gen:4d}/{n_gen}  |  nd={nd_mask.sum():3d}  |  HV={hv_val:.4f}")

        # Environmental selection → new archive
        archive, raw_archive, pen_archive = _environmental_selection(
            combined, raw_combined, pen_combined, fitness, archive_size
        )

        # Mating selection from archive (binary tournament, lower F = better)
        arch_fitness  = _assign_fitness(pen_archive)
        is_better     = lambda i, j: arch_fitness[i] < arch_fitness[j]
        parent_idx    = binary_tournament(len(archive), pop_size, is_better)
        parents       = archive[parent_idx]

        # Offspring
        pop = _make_offspring(parents, p_c)
        raw_pop, pen_pop = _evaluate(pop, ds, lam, alpha)

    # Final HV
    combined     = np.concatenate([pop, archive], axis=0)
    raw_combined = np.concatenate([raw_pop, raw_archive], axis=0)
    pen_combined = np.concatenate([pen_pop, pen_archive], axis=0)
    fitness      = _assign_fitness(pen_combined)
    nd_mask      = fitness < 1.0
    hv_history.append(
        hypervolume(raw_combined[nd_mask], hv_ref) if nd_mask.any() else 0.0
    )

    # Pareto front: non-dominated solutions (raw objectives)
    pareto_front = raw_combined[nd_mask]

    return SPEA2Result(
        population=pop,
        archive=archive,
        objectives=raw_pop,
        archive_obj=raw_archive,
        fitness=fitness,
        pareto_front=pareto_front,
        hv_history=hv_history,
    )


# ── Fitness assignment ────────────────────────────────────────────────────────

def _assign_fitness(objectives: np.ndarray) -> np.ndarray:
    """
    Compute SPEA2 fitness F(i) = R(i) + D(i) for each individual.

    Operates on penalised objectives (lower = better for all).
    """
    n = len(objectives)
    if n == 0:
        return np.empty(0)

    # Pairwise dominance: dom[i,j] = True iff i dominates j
    diff = objectives[:, np.newaxis, :] - objectives[np.newaxis, :, :]  # (n,n,3)
    dom  = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
    np.fill_diagonal(dom, False)

    strength = dom.sum(axis=1).astype(float)    # S(i)
    raw_fit  = dom.T @ strength                  # R(i) = sum S(j) for j dominating i

    # k-NN density in penalised objective space
    k = max(1, int(np.sqrt(n)))
    dist_matrix = _pairwise_distances(objectives)
    np.fill_diagonal(dist_matrix, np.inf)
    sorted_dists = np.sort(dist_matrix, axis=1)   # (n, n-1) ascending
    sigma_k = sorted_dists[:, k - 1]              # k-th nearest distance
    density  = 1.0 / (sigma_k + 2.0)

    return raw_fit + density


def _pairwise_distances(obj: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix, shape (n, n)."""
    diff = obj[:, np.newaxis, :] - obj[np.newaxis, :, :]  # (n,n,m)
    return np.sqrt((diff ** 2).sum(axis=2))


# ── Environmental selection ───────────────────────────────────────────────────

def _environmental_selection(
    combined:     np.ndarray,
    raw_combined: np.ndarray,
    pen_combined: np.ndarray,
    fitness:      np.ndarray,
    archive_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill archive of size `archive_size` from combined pool.

    1. All F < 1 (non-dominated) enter unconditionally.
    2. Too few → add best dominated (ascending F).
    3. Too many → truncate by iterative minimum-distance pruning.
    """
    nd_idx  = np.where(fitness < 1.0)[0].tolist()
    dom_idx = np.where(fitness >= 1.0)[0].tolist()

    if len(nd_idx) == archive_size:
        sel = nd_idx

    elif len(nd_idx) < archive_size:
        # Fill remainder from dominated, sorted by F ascending
        needed   = archive_size - len(nd_idx)
        dom_arr  = np.array(dom_idx)
        order    = np.argsort(fitness[dom_arr])
        fill_idx = dom_arr[order[:needed]].tolist()
        sel      = nd_idx + fill_idx

    else:
        # Too many non-dominated → truncate
        sel = _truncate(nd_idx, pen_combined, archive_size)

    sel = np.array(sel)
    return combined[sel], raw_combined[sel], pen_combined[sel]


def _truncate(
    indices: list[int],
    objectives: np.ndarray,
    target: int,
) -> list[int]:
    """
    Iteratively remove the individual with the smallest distance to its nearest
    neighbour. Ties broken by comparing next-nearest distances.
    Returns exactly `target` indices.
    """
    remaining = list(indices)

    while len(remaining) > target:
        pts = objectives[remaining]             # (k, 3)
        d   = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        d_sorted = np.sort(d, axis=1)           # each row sorted ascending

        # Lexicographic minimum across rows → pick the worst individual
        # (the one whose sorted distance vector is smallest)
        min_row = 0
        for i in range(1, len(remaining)):
            for col in range(d_sorted.shape[1]):
                if d_sorted[i, col] < d_sorted[min_row, col]:
                    min_row = i
                    break
                elif d_sorted[i, col] > d_sorted[min_row, col]:
                    break

        remaining.pop(min_row)

    return remaining


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(
    pop: np.ndarray,
    ds: DataStore,
    lam: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    n   = len(pop)
    raw = np.empty((n, 3))
    pen = np.empty((n, 3))

    for i, chrom in enumerate(pop):
        menu    = decode(chrom, ds)
        obj     = compute_objectives(menu, ds)
        R, _, _ = compute_penalty(menu, ds, alpha)
        raw[i]  = obj
        pen[i]  = apply_penalty(obj, R, lam)

    return raw, pen


# ── Offspring creation ────────────────────────────────────────────────────────

def _make_offspring(parents: np.ndarray, p_c: float) -> np.ndarray:
    n_parents = len(parents)
    offspring = np.empty_like(parents)

    for i in range(0, n_parents - 1, 2):
        c1, c2 = crossover(parents[i], parents[i + 1], p_c)
        offspring[i]     = mutate(c1)
        offspring[i + 1] = mutate(c2)

    if n_parents % 2 == 1:
        offspring[-1] = mutate(parents[-1].copy())

    return offspring
