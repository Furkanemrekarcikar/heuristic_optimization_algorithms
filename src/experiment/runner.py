"""
Experiment runner for the MODP project.

Three experiments:
  1. User comparison     — NSGA-II on User1 vs User2
  2. Algorithm comparison — NSGA-II vs SPEA2 on User1 (repeated on User2)
  3. Diversity impact    — NSGA-II with alpha=0 vs alpha=0.5 on User1

Each run produces:
  - pareto_front.csv     (raw objectives of the final non-dominated set)
  - hv_history.csv       (hypervolume per generation, using shared reference)
  - best_menus.csv       (food items for 3 representative solutions)
  - nutrient_summary.csv (nutrient totals vs DRI bounds per solution)
  - run_summary.json     (scalar metrics incl. shared_hv_ref values)

Hypervolume reference point:
  Per the project handout, all algorithms in an experiment use the SAME
  reference point: worst objective value observed across all runs + 10% margin.
  The reference point is estimated before any run starts via random sampling,
  then passed to every algorithm call so convergence curves are comparable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..database.connection import DBConnection
from ..database.loader import DataStore, load_data
from ..problem.chromosome import random_chromosome, decode
from ..problem.objectives import compute_objectives
from ..problem.penalty import compute_penalty, apply_penalty
from ..algorithms import nsga2, spea2
from .. import config as cfg


Algorithm = Literal["nsga2", "spea2"]


@dataclass
class RunConfig:
    user_id:      int
    algorithm:    Algorithm
    pop_size:     int   = cfg.POP_SIZE
    n_gen:        int   = cfg.N_GEN
    p_c:          float = cfg.P_CROSS
    lam:          float = cfg.LAMBDA_PENALTY
    alpha:        float = cfg.ALPHA_DIVERSITY
    label:        str   = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"user{self.user_id}_{self.algorithm}_a{self.alpha:.2f}"


@dataclass
class RunResult:
    label:        str
    pareto_front: np.ndarray   # (n, 3) raw objectives
    hv_history:   list[float]
    best_menus:   pd.DataFrame
    nutrient_summary: pd.DataFrame
    summary:      dict
    ds:           DataStore


# ── Public interface ──────────────────────────────────────────────────────────

def estimate_shared_hv_ref(
    ds_list: list[DataStore],
    n_samples: int = 100,
) -> np.ndarray:
    """
    Estimate a shared HV reference point for an experiment.

    Samples `n_samples` random chromosomes from each DataStore, evaluates raw
    objectives, and returns  max_over_all_samples * 1.1 + 1e-6.

    Using the same reference point across all algorithms ensures HV values
    are directly comparable (required by the project handout).
    """
    all_obj: list[np.ndarray] = []
    for ds in ds_list:
        for _ in range(n_samples):
            chrom = random_chromosome(ds.n_foods)
            menu  = decode(chrom, ds)
            all_obj.append(compute_objectives(menu, ds))
    arr = np.array(all_obj)
    return arr.max(axis=0) * 1.1 + 1e-6


def run_single(
    config: RunConfig,
    ds: DataStore,
    hv_ref: np.ndarray | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> RunResult:
    """
    Execute one algorithm run and optionally save results to `out_dir`.

    Parameters
    ----------
    config  : RunConfig
    ds      : DataStore for the target user
    hv_ref  : shared HV reference point — pass the SAME array to all runs
              in an experiment so HV values are comparable. If None, the
              algorithm will auto-estimate (not recommended for comparisons).
    out_dir : save results here if given
    verbose : print progress
    """
    if verbose:
        print(f"\n[{config.label}]  pop={config.pop_size}  gen={config.n_gen}"
              f"  alg={config.algorithm}  alpha={config.alpha}")

    t0 = time.time()

    if config.algorithm == "nsga2":
        result = nsga2.run(
            ds,
            pop_size=config.pop_size,
            n_gen=config.n_gen,
            p_c=config.p_c,
            lam=config.lam,
            alpha=config.alpha,
            hv_ref=hv_ref,
            verbose=verbose,
        )
        pareto_front  = result.pareto_front
        hv_history    = result.hv_history
        front0_mask   = result.ranks == 0
        front0_chroms = result.population[front0_mask]
        front0_raw    = result.objectives[front0_mask]

    elif config.algorithm == "spea2":
        result = spea2.run(
            ds,
            pop_size=config.pop_size,
            archive_size=config.pop_size,
            n_gen=config.n_gen,
            p_c=config.p_c,
            lam=config.lam,
            alpha=config.alpha,
            hv_ref=hv_ref,
            verbose=verbose,
        )
        pareto_front  = result.pareto_front
        hv_history    = result.hv_history
        front0_chroms = result.archive
        front0_raw    = result.archive_obj

    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    elapsed    = time.time() - t0
    final_hv   = hv_history[-1] if hv_history else 0.0
    front_size = len(pareto_front)

    summary = {
        "label":       config.label,
        "algorithm":   config.algorithm,
        "user_id":     config.user_id,
        "pop_size":    config.pop_size,
        "n_gen":       config.n_gen,
        "alpha":       config.alpha,
        "final_hv":    float(final_hv),
        "front_size":  int(front_size),
        "elapsed_s":   round(elapsed, 2),
        "hv_ref":      hv_ref.tolist() if hv_ref is not None else None,
    }

    best_df    = _best_menu_table(front0_chroms, front0_raw, ds)
    nutr_df    = _nutrient_summary_table(front0_chroms, front0_raw, ds)

    if out_dir is not None:
        _save_results(
            out_dir      = Path(out_dir) / config.label,
            pareto       = pareto_front,
            hv_history   = hv_history,
            best_df      = best_df,
            nutr_df      = nutr_df,
            summary      = summary,
        )

    return RunResult(
        label            = config.label,
        pareto_front     = pareto_front,
        hv_history       = hv_history,
        best_menus       = best_df,
        nutrient_summary = nutr_df,
        summary          = summary,
        ds               = ds,
    )


def load_ds_for(user_id: int) -> DataStore:
    """Open DB connection and load DataStore."""
    db = DBConnection(
        cfg.DB_HOST, cfg.DB_PORT, cfg.DB_USER, cfg.DB_PASSWORD, cfg.DB_NAME
    )
    return load_data(db, user_id)


# ── Experiment definitions ────────────────────────────────────────────────────

def experiment_1_user_comparison(
    out_dir: Path,
    pop_size: int = cfg.POP_SIZE,
    n_gen: int    = cfg.N_GEN,
    verbose: bool = True,
) -> list[RunResult]:
    """Exp 1: NSGA-II on User1 vs User2 — shared HV reference point."""
    ds1 = load_ds_for(cfg.USER1_ID)
    ds2 = load_ds_for(cfg.USER2_ID)

    hv_ref = estimate_shared_hv_ref([ds1, ds2])
    if verbose:
        print(f"\n[Exp 1] Shared HV ref: {hv_ref.round(4)}")

    results = []
    for uid, ds in [(cfg.USER1_ID, ds1), (cfg.USER2_ID, ds2)]:
        cfg_run = RunConfig(
            user_id   = uid,
            algorithm = "nsga2",
            pop_size  = pop_size,
            n_gen     = n_gen,
            label     = f"exp1_user{uid}_nsga2",
        )
        results.append(run_single(cfg_run, ds, hv_ref=hv_ref, out_dir=out_dir, verbose=verbose))
    return results


def experiment_2_algorithm_comparison(
    out_dir: Path,
    pop_size: int = cfg.POP_SIZE,
    n_gen: int    = cfg.N_GEN,
    verbose: bool = True,
) -> list[RunResult]:
    """Exp 2: NSGA-II vs SPEA2 on both users — shared HV reference point."""
    ds1 = load_ds_for(cfg.USER1_ID)
    ds2 = load_ds_for(cfg.USER2_ID)

    hv_ref = estimate_shared_hv_ref([ds1, ds2])
    if verbose:
        print(f"\n[Exp 2] Shared HV ref: {hv_ref.round(4)}")

    results = []
    for uid, ds in [(cfg.USER1_ID, ds1), (cfg.USER2_ID, ds2)]:
        for alg in ("nsga2", "spea2"):
            cfg_run = RunConfig(
                user_id   = uid,
                algorithm = alg,
                pop_size  = pop_size,
                n_gen     = n_gen,
                label     = f"exp2_user{uid}_{alg}",
            )
            results.append(run_single(cfg_run, ds, hv_ref=hv_ref, out_dir=out_dir, verbose=verbose))
    return results


def experiment_3_diversity_impact(
    out_dir: Path,
    pop_size: int = cfg.POP_SIZE,
    n_gen: int    = cfg.N_GEN,
    verbose: bool = True,
) -> list[RunResult]:
    """Exp 3: NSGA-II with alpha=0 vs alpha=0.5 on both users — shared HV reference."""
    ds1 = load_ds_for(cfg.USER1_ID)
    ds2 = load_ds_for(cfg.USER2_ID)

    hv_ref = estimate_shared_hv_ref([ds1, ds2])
    if verbose:
        print(f"\n[Exp 3] Shared HV ref: {hv_ref.round(4)}")

    results = []
    for uid, ds in [(cfg.USER1_ID, ds1), (cfg.USER2_ID, ds2)]:
        for alpha in (0.0, cfg.ALPHA_DIVERSITY):
            cfg_run = RunConfig(
                user_id   = uid,
                algorithm = "nsga2",
                pop_size  = pop_size,
                n_gen     = n_gen,
                alpha     = alpha,
                label     = f"exp3_user{uid}_alpha{alpha:.2f}",
            )
            results.append(run_single(cfg_run, ds, hv_ref=hv_ref, out_dir=out_dir, verbose=verbose))
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _best_menu_table(
    chromosomes: np.ndarray,
    objectives:  np.ndarray,
    ds:          DataStore,
) -> pd.DataFrame:
    """
    3 representative solutions (most preferred, cheapest, least CO2).
    One row per food item, with solution-level objective totals appended.
    """
    if len(chromosomes) == 0:
        return pd.DataFrame()

    rows     = []
    criteria = [("Most Preferred", 0), ("Cheapest", 1), ("Least CO2", 2)]
    seen     = set()

    for crit_name, obj_col in criteria:
        idx = int(np.argmin(objectives[:, obj_col]))
        if idx in seen:
            continue
        seen.add(idx)

        menu          = decode(chromosomes[idx], ds)
        selected_mask = menu.breakfast_mask | menu.lunch_dinner_mask
        n_groups      = len(np.unique(ds.food_group_ids[selected_mask])) if selected_mask.any() else 0

        for food_idx in np.where(selected_mask)[0]:
            meal = "Breakfast" if menu.breakfast_mask[food_idx] else "Lunch/Dinner"
            rows.append({
                "solution":          crit_name,
                "meal":              meal,
                "food_id":           int(ds.food_ids[food_idx]),
                "food_name":         ds.food_names[food_idx],
                "preference":        float(ds.preferences[food_idx]),
                "cost":              float(ds.costs[food_idx]),
                "co2":               float(ds.co2s[food_idx]),
                "preference_total":  float(-objectives[idx, 0]),
                "cost_total":        float(objectives[idx, 1]),
                "co2_total":         float(objectives[idx, 2]),
                "distinct_groups":   n_groups,
            })

    return pd.DataFrame(rows)


def _nutrient_summary_table(
    chromosomes: np.ndarray,
    objectives:  np.ndarray,
    ds:          DataStore,
) -> pd.DataFrame:
    """
    Nutrient totals vs DRI bounds for the 3 representative solutions.
    One row per (solution, nutrient).
    """
    if len(chromosomes) == 0:
        return pd.DataFrame()

    nutrient_names = [
        cfg.NUTRIENT_NAMES.get(nid, f"Nutrient {nid}")
        for nid in cfg.CONSTRAINT_NUTRIENT_IDS
    ]
    rows     = []
    criteria = [("Most Preferred", 0), ("Cheapest", 1), ("Least CO2", 2)]
    seen     = set()

    for crit_name, obj_col in criteria:
        idx = int(np.argmin(objectives[:, obj_col]))
        if idx in seen:
            continue
        seen.add(idx)

        menu = decode(chromosomes[idx], ds)

        for j, name in enumerate(nutrient_names):
            total = float(menu.nutrient_totals[j])
            rll   = float(ds.dri_rll[j])
            rul   = float(ds.dri_rul[j])
            rows.append({
                "solution":       crit_name,
                "nutrient":       name,
                "total":          round(total, 2),
                "RLL":            round(rll,   2),
                "RUL":            round(rul,   2),
                "within_bounds":  rll <= total <= rul,
                "pct_of_RLL":     round(total / rll * 100, 1) if rll > 0 else None,
            })

    return pd.DataFrame(rows)


def _save_results(
    out_dir:    Path,
    pareto:     np.ndarray,
    hv_history: list[float],
    best_df:    pd.DataFrame,
    nutr_df:    pd.DataFrame,
    summary:    dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        pareto, columns=["neg_preference", "cost", "co2"]
    ).to_csv(out_dir / "pareto_front.csv", index=False)

    pd.DataFrame(
        {"generation": range(len(hv_history)), "hypervolume": hv_history}
    ).to_csv(out_dir / "hv_history.csv", index=False)

    if not best_df.empty:
        best_df.to_csv(out_dir / "best_menus.csv", index=False)

    if not nutr_df.empty:
        nutr_df.to_csv(out_dir / "nutrient_summary.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
