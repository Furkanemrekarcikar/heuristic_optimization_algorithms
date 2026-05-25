"""
Run all 3 MODP experiments and save results + plots.

Usage:
    python scripts/run_experiments.py           # full run (pop=100, gen=200)
    python scripts/run_experiments.py --quick   # smoke run (pop=20, gen=10)
    python scripts/run_experiments.py --exp 1   # single experiment

Outputs go to results/<exp_name>/<run_label>/
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import src.config as cfg
from src.experiment.runner import (
    experiment_1_user_comparison,
    experiment_2_algorithm_comparison,
    experiment_3_diversity_impact,
)
from src.visualization.plots import (
    pareto_plot,
    convergence_plot,
    menu_table_plot,
    summary_bar_plot,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def make_plots_exp1(results, out_dir: Path):
    """Exp 1: User1 vs User2 — Pareto fronts + convergence + menus."""
    fronts     = {r.label: r.pareto_front for r in results}
    histories  = {r.label: r.hv_history   for r in results}
    summaries  = [r.summary               for r in results]

    pareto_plot(
        fronts,
        x_col=1, y_col=0,
        title="Exp 1: User Comparison — Cost vs Preference",
        out_path=out_dir / "pareto_cost_vs_pref.png",
    )
    pareto_plot(
        fronts,
        x_col=2, y_col=0,
        title="Exp 1: User Comparison — CO2 vs Preference",
        out_path=out_dir / "pareto_co2_vs_pref.png",
    )
    convergence_plot(
        histories,
        title="Exp 1: HV Convergence — User1 vs User2",
        out_path=out_dir / "convergence.png",
    )
    summary_bar_plot(
        summaries,
        metric="final_hv",
        title="Exp 1: Final Hypervolume",
        out_path=out_dir / "hv_bar.png",
    )

    for r in results:
        if not r.best_menus.empty:
            menu_table_plot(
                r.best_menus,
                solution_label="Most Preferred",
                title=f"Best Menu — {r.label}",
                out_path=out_dir / f"menu_{r.label}.png",
            )


def make_plots_exp2(results, out_dir: Path):
    """Exp 2: NSGA-II vs SPEA2 — per user."""
    import itertools
    user_ids = list(dict.fromkeys(r.summary["user_id"] for r in results))

    for uid in user_ids:
        uid_results = [r for r in results if r.summary["user_id"] == uid]
        fronts      = {r.label: r.pareto_front for r in uid_results}
        histories   = {r.label: r.hv_history   for r in uid_results}

        pareto_plot(
            fronts,
            x_col=1, y_col=0,
            title=f"Exp 2: Algorithm Comparison (User {uid}) — Cost vs Preference",
            out_path=out_dir / f"pareto_user{uid}.png",
        )
        convergence_plot(
            histories,
            title=f"Exp 2: HV Convergence (User {uid})",
            out_path=out_dir / f"convergence_user{uid}.png",
        )

    summary_bar_plot(
        [r.summary for r in results],
        metric="final_hv",
        title="Exp 2: Final Hypervolume — Algorithm Comparison",
        out_path=out_dir / "hv_bar.png",
    )


def make_plots_exp3(results, out_dir: Path):
    """Exp 3: Diversity impact — alpha=0 vs alpha=0.5."""
    user_ids = list(dict.fromkeys(r.summary["user_id"] for r in results))

    for uid in user_ids:
        uid_results = [r for r in results if r.summary["user_id"] == uid]
        fronts      = {r.label: r.pareto_front for r in uid_results}
        histories   = {r.label: r.hv_history   for r in uid_results}

        pareto_plot(
            fronts,
            x_col=1, y_col=0,
            title=f"Exp 3: Diversity Impact (User {uid}) — Cost vs Preference",
            out_path=out_dir / f"pareto_user{uid}.png",
        )
        convergence_plot(
            histories,
            title=f"Exp 3: HV Convergence (User {uid})",
            out_path=out_dir / f"convergence_user{uid}.png",
        )

    summary_bar_plot(
        [r.summary for r in results],
        metric="final_hv",
        title="Exp 3: Diversity Impact — Final Hypervolume",
        out_path=out_dir / "hv_bar.png",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MODP experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke run: pop=20, gen=10")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3],
                        help="Run only this experiment (1, 2, or 3)")
    args = parser.parse_args()

    pop  = 20  if args.quick else cfg.POP_SIZE
    gen  = 10  if args.quick else cfg.N_GEN

    print(f"{'='*60}")
    print(f"  MODP Experiments  (pop={pop}, gen={gen})")
    print(f"  Results -> {RESULTS_DIR}")
    print(f"{'='*60}")

    run_exp = args.exp  # None means run all

    if run_exp in (None, 1):
        print("\n### Experiment 1: User Comparison ###")
        out1 = RESULTS_DIR / "exp1_user_comparison"
        r1   = experiment_1_user_comparison(out1, pop_size=pop, n_gen=gen)
        make_plots_exp1(r1, out1)
        print(f"  Saved -> {out1}")

    if run_exp in (None, 2):
        print("\n### Experiment 2: Algorithm Comparison ###")
        out2 = RESULTS_DIR / "exp2_algorithm_comparison"
        r2   = experiment_2_algorithm_comparison(out2, pop_size=pop, n_gen=gen)
        make_plots_exp2(r2, out2)
        print(f"  Saved -> {out2}")

    if run_exp in (None, 3):
        print("\n### Experiment 3: Diversity Impact ###")
        out3 = RESULTS_DIR / "exp3_diversity_impact"
        r3   = experiment_3_diversity_impact(out3, pop_size=pop, n_gen=gen)
        make_plots_exp3(r3, out3)
        print(f"  Saved -> {out3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
