"""
NSGA-II for the Multi-Objective Diet Problem (MODP).

Objectives (minimization form):
    obj[0] = -preference   (maximise preference → minimise negative)
    obj[1] =  cost
    obj[2] =  co2

Selection criterion (Deb et al., 2002):
    i ≺ j  iff  rank[i] < rank[j]
              OR (rank[i] == rank[j] AND crowd[i] > crowd[j])
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

class NSGAIIResult(NamedTuple):
    population:    np.ndarray   # shape (pop_size, n_foods)  — final chromosomes
    objectives:    np.ndarray   # shape (pop_size, 3)        — raw objectives
    pen_objectives: np.ndarray  # shape (pop_size, 3)        — penalised objectives
    ranks:         np.ndarray   # shape (pop_size,)          — Pareto rank (0 = best)
    crowding:      np.ndarray   # shape (pop_size,)
    pareto_front:  np.ndarray   # shape (n_front0, 3)        — rank-0 raw objectives
    hv_history:    list[float]  # hypervolume per generation


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    ds: DataStore,
    pop_size: int     = cfg.POP_SIZE,
    n_gen: int        = cfg.N_GEN,
    p_c: float        = cfg.P_CROSS,
    lam: float        = cfg.LAMBDA_PENALTY,
    alpha: float      = cfg.ALPHA_DIVERSITY,
    hv_ref: np.ndarray | None = None,
    verbose: bool     = True,
) -> NSGAIIResult:
    """
    Run NSGA-II.

    Parameters
    ----------
    ds       : DataStore loaded for the target user
    pop_size : population size
    n_gen    : number of generations
    p_c      : crossover probability
    lam      : penalty weight λ
    alpha    : diversity penalty weight α
    hv_ref   : reference point for hypervolume tracking (shape (3,)).
               If None, auto-set from initial population.
    verbose  : print progress every 10 generations
    """
    n = ds.food_ids.shape[0]

    # ── Initialise ────────────────────────────────────────────────────────────
    pop = np.array([random_chromosome(n) for _ in range(pop_size)])
    raw_obj, pen_obj = _evaluate(pop, ds, lam, alpha)

    ranks, crowding = _rank_and_crowd(pen_obj)

    if hv_ref is None:
        hv_ref = raw_obj.max(axis=0) * 1.1 + 1e-6

    hv_history: list[float] = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for gen in range(n_gen):
        # Record HV of current non-dominated front
        front0_mask = ranks == 0
        hv_val = hypervolume(raw_obj[front0_mask], hv_ref) if front0_mask.any() else 0.0
        hv_history.append(hv_val)

        if verbose and gen % 10 == 0:
            n_front0 = front0_mask.sum()
            print(f"  Gen {gen:4d}/{n_gen}  |  front0={n_front0:3d}  |  HV={hv_val:.4f}")

        # Selection: 2×pop_size parents
        is_better = _make_comparator(ranks, crowding)
        parents_idx = binary_tournament(pop_size, pop_size, is_better)

        # Offspring
        offspring = _make_offspring(pop[parents_idx], p_c)
        off_raw, off_pen = _evaluate(offspring, ds, lam, alpha)

        # Combine parent + offspring pools
        comb_pop = np.concatenate([pop, offspring], axis=0)
        comb_raw = np.concatenate([raw_obj, off_raw], axis=0)
        comb_pen = np.concatenate([pen_obj, off_pen], axis=0)

        # Reduce back to pop_size
        pop, raw_obj, pen_obj = _select_next(
            comb_pop, comb_raw, comb_pen, pop_size
        )
        ranks, crowding = _rank_and_crowd(pen_obj)

    # Final HV
    front0_mask = ranks == 0
    hv_history.append(
        hypervolume(raw_obj[front0_mask], hv_ref) if front0_mask.any() else 0.0
    )

    pareto_front = raw_obj[front0_mask]

    return NSGAIIResult(
        population=pop,
        objectives=raw_obj,
        pen_objectives=pen_obj,
        ranks=ranks,
        crowding=crowding,
        pareto_front=pareto_front,
        hv_history=hv_history,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(
    pop: np.ndarray,
    ds: DataStore,
    lam: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_objectives, penalised_objectives) for every chromosome."""
    n = len(pop)
    raw = np.empty((n, 3))
    pen = np.empty((n, 3))

    for i, chrom in enumerate(pop):
        menu       = decode(chrom, ds)
        obj        = compute_objectives(menu, ds)
        R, _, _    = compute_penalty(menu, ds, alpha)
        raw[i]     = obj
        pen[i]     = apply_penalty(obj, R, lam)

    return raw, pen


# ── Non-dominated sorting + crowding distance ─────────────────────────────────

def _rank_and_crowd(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks    = _fast_nondominated_sort(objectives)
    crowding = _crowding_distance(objectives, ranks)
    return ranks, crowding


def _fast_nondominated_sort(objectives: np.ndarray) -> np.ndarray:
    """
    Vectorised fast non-dominated sorting.

    Returns integer rank array (0 = Pareto front 1, 1 = front 2, …).
    Uses pairwise dominance broadcast: O(n² × m) time, O(n²) memory.
    Acceptable for pop_size ≤ 500.
    """
    n   = len(objectives)
    obj = objectives         # (n, 3)

    # dom[i, j] = True iff i dominates j  (all i <= j AND at least one i < j)
    diff = obj[:, np.newaxis, :] - obj[np.newaxis, :, :]  # (n, n, 3)
    dom  = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)  # (n, n)
    np.fill_diagonal(dom, False)

    ranks      = np.full(n, -1, dtype=int)
    dominated  = dom.sum(axis=0)   # how many solutions dominate each i
    dominate_s = [np.where(dom[i])[0].tolist() for i in range(n)]

    current_rank = 0
    current_front = np.where(dominated == 0)[0].tolist()

    while current_front:
        for i in current_front:
            ranks[i] = current_rank
        next_front: list[int] = []
        for i in current_front:
            for j in dominate_s[i]:
                dominated[j] -= 1
                if dominated[j] == 0:
                    next_front.append(j)
        current_front = next_front
        current_rank += 1

    return ranks


def _crowding_distance(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """
    Crowding distance per individual, computed within each Pareto front.
    Boundary solutions in each front get distance = inf.
    """
    n        = len(objectives)
    distance = np.zeros(n)

    for r in range(ranks.max() + 1):
        idx = np.where(ranks == r)[0]
        if len(idx) <= 2:
            distance[idx] = np.inf
            continue

        front_obj = objectives[idx]       # (k, 3)
        k         = len(idx)
        d         = np.zeros(k)

        for m in range(objectives.shape[1]):
            order   = np.argsort(front_obj[:, m])
            f_min   = front_obj[order[0],  m]
            f_max   = front_obj[order[-1], m]
            span    = f_max - f_min if f_max > f_min else 1.0

            d[order[0]]  = np.inf
            d[order[-1]] = np.inf

            for pos in range(1, k - 1):
                d[order[pos]] += (
                    front_obj[order[pos + 1], m] - front_obj[order[pos - 1], m]
                ) / span

        distance[idx] = d

    return distance


# ── Selection of next generation ──────────────────────────────────────────────

def _select_next(
    comb_pop: np.ndarray,
    comb_raw: np.ndarray,
    comb_pen: np.ndarray,
    pop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill next generation front-by-front, using crowding distance to break ties
    on the last front that doesn't fit entirely.
    """
    ranks, crowding = _rank_and_crowd(comb_pen)

    selected: list[int] = []
    r = 0
    while True:
        front_idx = np.where(ranks == r)[0].tolist()
        if len(selected) + len(front_idx) <= pop_size:
            selected.extend(front_idx)
            if len(selected) == pop_size:
                break
            r += 1
        else:
            # Fill remaining slots by crowding distance (descending)
            needed  = pop_size - len(selected)
            cd      = crowding[front_idx]
            top_idx = np.argsort(-cd)[:needed]
            selected.extend([front_idx[i] for i in top_idx])
            break

    sel = np.array(selected)
    return comb_pop[sel], comb_raw[sel], comb_pen[sel]


# ── Offspring creation ────────────────────────────────────────────────────────

def _make_offspring(parents: np.ndarray, p_c: float) -> np.ndarray:
    """
    Pair up parents sequentially; apply OX crossover + swap mutation.
    Returns offspring array of same length as parents.
    """
    n_parents = len(parents)
    offspring = np.empty_like(parents)

    for i in range(0, n_parents - 1, 2):
        c1, c2 = crossover(parents[i], parents[i + 1], p_c)
        offspring[i]     = mutate(c1)
        offspring[i + 1] = mutate(c2)

    if n_parents % 2 == 1:
        offspring[-1] = mutate(parents[-1].copy())

    return offspring


# ── Comparator factory ────────────────────────────────────────────────────────

def _make_comparator(ranks: np.ndarray, crowding: np.ndarray):
    """Return is_better(i, j) suitable for binary_tournament."""
    def is_better(i: int, j: int) -> bool:
        if ranks[i] < ranks[j]:
            return True
        if ranks[i] == ranks[j] and crowding[i] > crowding[j]:
            return True
        return False
    return is_better
