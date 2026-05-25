"""
Experiment runner for the MODP project.

Three experiments:
  1. User comparison     — NSGA-II on User1 vs User2
  2. Algorithm comparison — NSGA-II vs SPEA2 on User1 (repeated on User2)
  3. Diversity impact    — NSGA-II with alpha=0 vs alpha=0.5 on User1

Each run produces:
  - pareto_front.csv   (raw objectives of the final non-dominated set)
  - hv_history.csv     (hypervolume per generation)
  - best_menus.csv     (top-3 solutions: most preferred, cheapest, least CO2)
  - run_summary.json   (scalar metrics)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..database.connection import DBConnection
from ..database.loader import DataStore, load_data
from ..problem.chromosome import decode
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
    label:        str   = ""   # human-readable tag, auto-set if empty

    def __post_init__(self):
        if not self.label:
            self.label = f"user{self.user_id}_{self.algorithm}_a{self.alpha:.2f}"


@dataclass
class RunResult:
    label:        str
    pareto_front: np.ndarray   # (n, 3) raw objectives
    hv_history:   list[float]
    best_menus:   pd.DataFrame
    summary:      dict
    ds:           DataStore    # kept for downstream use (not serialised)


# ── Public interface ──────────────────────────────────────────────────────────

def run_single(
    config: RunConfig,
    ds: DataStore,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> RunResult:
    """
    Execute one algorithm run and optionally save results to `out_dir`.
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
            verbose=verbose,
        )
        pareto_front = result.pareto_front
        hv_history   = result.hv_history

        # Best chromosomes: rank-0 individuals
        front0_mask  = result.ranks == 0
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
            verbose=verbose,
        )
        pareto_front  = result.pareto_front
        hv_history    = result.hv_history
        front0_chroms = result.archive
        front0_raw    = result.archive_obj

    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    elapsed = time.time() - t0

    # ── Derive scalar summary ─────────────────────────────────────────────────
    final_hv  = hv_history[-1] if hv_history else 0.0
    front_size = len(pareto_front)

    summary = {
        "label":      config.label,
        "algorithm":  config.algorithm,
        "user_id":    config.user_id,
        "pop_size":   config.pop_size,
        "n_gen":      config.n_gen,
        "alpha":      config.alpha,
        "final_hv":   float(final_hv),
        "front_size": int(front_size),
        "elapsed_s":  round(elapsed, 2),
    }

    # ── Build best-menu table ─────────────────────────────────────────────────
    best_df = _best_menu_table(front0_chroms, front0_raw, ds)

    # ── Save to disk ──────────────────────────────────────────────────────────
    if out_dir is not None:
        _save_results(
            out_dir    = Path(out_dir) / config.label,
            pareto     = pareto_front,
            hv_history = hv_history,
            best_df    = best_df,
            summary    = summary,
        )

    return RunResult(
        label        = config.label,
        pareto_front = pareto_front,
        hv_history   = hv_history,
        best_menus   = best_df,
        summary      = summary,
        ds           = ds,
    )


def load_ds_for(user_id: int) -> DataStore:
    """Convenience: open DB connection and load DataStore."""
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
    """Exp 1: NSGA-II on User1 vs User2 (same parameters)."""
    results = []
    for uid in [cfg.USER1_ID, cfg.USER2_ID]:
        ds  = load_ds_for(uid)
        cfg_run = RunConfig(
            user_id   = uid,
            algorithm = "nsga2",
            pop_size  = pop_size,
            n_gen     = n_gen,
            label     = f"exp1_user{uid}_nsga2",
        )
        results.append(run_single(cfg_run, ds, out_dir=out_dir, verbose=verbose))
    return results


def experiment_2_algorithm_comparison(
    out_dir: Path,
    pop_size: int = cfg.POP_SIZE,
    n_gen: int    = cfg.N_GEN,
    verbose: bool = True,
) -> list[RunResult]:
    """Exp 2: NSGA-II vs SPEA2 on both users."""
    results = []
    for uid in [cfg.USER1_ID, cfg.USER2_ID]:
        ds = load_ds_for(uid)
        for alg in ("nsga2", "spea2"):
            cfg_run = RunConfig(
                user_id   = uid,
                algorithm = alg,
                pop_size  = pop_size,
                n_gen     = n_gen,
                label     = f"exp2_user{uid}_{alg}",
            )
            results.append(run_single(cfg_run, ds, out_dir=out_dir, verbose=verbose))
    return results


def experiment_3_diversity_impact(
    out_dir: Path,
    pop_size: int = cfg.POP_SIZE,
    n_gen: int    = cfg.N_GEN,
    verbose: bool = True,
) -> list[RunResult]:
    """Exp 3: NSGA-II with alpha=0 vs alpha=0.5 on both users."""
    results = []
    for uid in [cfg.USER1_ID, cfg.USER2_ID]:
        ds = load_ds_for(uid)
        for alpha in (0.0, cfg.ALPHA_DIVERSITY):
            cfg_run = RunConfig(
                user_id   = uid,
                algorithm = "nsga2",
                pop_size  = pop_size,
                n_gen     = n_gen,
                alpha     = alpha,
                label     = f"exp3_user{uid}_alpha{alpha:.2f}",
            )
            results.append(run_single(cfg_run, ds, out_dir=out_dir, verbose=verbose))
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _best_menu_table(
    chromosomes: np.ndarray,
    objectives:  np.ndarray,
    ds:          DataStore,
) -> pd.DataFrame:
    """
    Select 3 representative solutions from the Pareto front and build a
    human-readable menu DataFrame.

    Representatives:
      - Most preferred  (lowest obj[0] = -preference)
      - Cheapest        (lowest obj[1])
      - Least CO2       (lowest obj[2])
    """
    if len(chromosomes) == 0:
        return pd.DataFrame()

    rows = []
    criteria = [
        ("Most Preferred", 0),
        ("Cheapest",       1),
        ("Least CO2",      2),
    ]
    seen = set()

    for crit_name, obj_col in criteria:
        idx = int(np.argmin(objectives[:, obj_col]))
        if idx in seen:
            continue
        seen.add(idx)

        chrom  = chromosomes[idx]
        menu   = decode(chrom, ds)
        selected_mask = menu.breakfast_mask | menu.lunch_dinner_mask
        sel_idx       = np.where(selected_mask)[0]

        for food_idx in sel_idx:
            meal = "Breakfast" if menu.breakfast_mask[food_idx] else "Lunch/Dinner"
            rows.append({
                "solution":   crit_name,
                "meal":       meal,
                "food_id":    int(ds.food_ids[food_idx]),
                "food_name":  ds.food_names[food_idx],
                "preference": float(ds.preferences[food_idx]),
                "cost":       float(ds.costs[food_idx]),
                "co2":        float(ds.co2s[food_idx]),
                "preference_total": float(-objectives[idx, 0]),
                "cost_total":       float(objectives[idx, 1]),
                "co2_total":        float(objectives[idx, 2]),
            })

    return pd.DataFrame(rows)


def _save_results(
    out_dir:    Path,
    pareto:     np.ndarray,
    hv_history: list[float],
    best_df:    pd.DataFrame,
    summary:    dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pareto front
    pd.DataFrame(
        pareto, columns=["neg_preference", "cost", "co2"]
    ).to_csv(out_dir / "pareto_front.csv", index=False)

    # HV history
    pd.DataFrame(
        {"generation": range(len(hv_history)), "hypervolume": hv_history}
    ).to_csv(out_dir / "hv_history.csv", index=False)

    # Best menus
    if not best_df.empty:
        best_df.to_csv(out_dir / "best_menus.csv", index=False)

    # Summary JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
