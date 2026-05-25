"""
Plotting utilities for the MODP project.

Functions:
  pareto_plot()      — 2-D scatter of Pareto fronts (obj1 vs obj2 or obj1 vs obj3)
  convergence_plot() — HV over generations for one or more runs
  menu_table_plot()  — Table figure showing best-menu food items
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Pareto front scatter ──────────────────────────────────────────────────────

def pareto_plot(
    fronts: dict[str, np.ndarray],
    x_col: int = 1,
    y_col: int = 0,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "Pareto Front",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Scatter plot of one or more Pareto fronts.

    Parameters
    ----------
    fronts   : {label: ndarray (n, 3)}  objectives in minimization form
    x_col    : column index for x-axis (default 1 = cost)
    y_col    : column index for y-axis (default 0 = -preference)
    x_label  : x-axis label (auto-derived from column if None)
    y_label  : y-axis label (auto-derived from column if None)
    title    : plot title
    out_path : save figure here if given
    """
    _OBJ_LABELS = {
        0: "Preference (negated)",
        1: "Cost",
        2: "CO₂ Footprint",
    }
    x_label = x_label or _OBJ_LABELS.get(x_col, f"Objective {x_col}")
    y_label = y_label or _OBJ_LABELS.get(y_col, f"Objective {y_col}")

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v"]

    for i, (label, front) in enumerate(fronts.items()):
        if len(front) == 0:
            continue
        ax.scatter(
            front[:, x_col],
            front[:, y_col],
            label=label,
            marker=markers[i % len(markers)],
            s=40,
            alpha=0.75,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Convergence (HV over generations) ────────────────────────────────────────

def convergence_plot(
    histories: dict[str, list[float]],
    title: str = "Hypervolume Convergence",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Line plot of hypervolume history for one or more runs.

    Parameters
    ----------
    histories : {label: [hv_gen0, hv_gen1, ...]}
    title     : plot title
    out_path  : save figure here if given
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    styles  = ["-", "--", "-.", ":"]

    for i, (label, hv) in enumerate(histories.items()):
        ax.plot(hv, label=label, linestyle=styles[i % len(styles)], linewidth=1.5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Menu table figure ─────────────────────────────────────────────────────────

def menu_table_plot(
    best_df: pd.DataFrame,
    solution_label: str = "Most Preferred",
    title: str = "Sample Daily Menu",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Render a selected solution's food items as a matplotlib table figure.

    Parameters
    ----------
    best_df         : DataFrame from runner._best_menu_table()
    solution_label  : which solution row to display
    title           : figure title
    out_path        : save figure here if given
    """
    sub = best_df[best_df["solution"] == solution_label].copy()
    if sub.empty:
        # Fall back to first available solution
        solution_label = best_df["solution"].iloc[0] if not best_df.empty else ""
        sub = best_df[best_df["solution"] == solution_label].copy()

    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        if out_path:
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
        return fig

    sub = sub[["meal", "food_name", "preference", "cost", "co2"]].reset_index(drop=True)
    sub.columns = ["Meal", "Food Item", "Preference", "Cost", "CO₂"]
    sub["Preference"] = sub["Preference"].map("{:.0f}".format)
    sub["Cost"]       = sub["Cost"].map("{:.2f}".format)
    sub["CO₂"]   = sub["CO₂"].map("{:.2f}".format)

    n_rows  = len(sub)
    fig_h   = max(3.0, 0.35 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText  = sub.values,
        colLabels = sub.columns,
        cellLoc   = "left",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(sub.columns))))

    # Header styling
    for col_idx in range(len(sub.columns)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for row_idx in range(1, n_rows + 1):
        colour = "#ecf0f1" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(sub.columns)):
            tbl[row_idx, col_idx].set_facecolor(colour)

    ax.set_title(f"{title}  ({solution_label})", fontsize=11, pad=12)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Summary bar chart ─────────────────────────────────────────────────────────

def summary_bar_plot(
    summaries: list[dict],
    metric: str = "final_hv",
    title: str = "Final Hypervolume Comparison",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Bar chart comparing a scalar metric across multiple runs.

    Parameters
    ----------
    summaries : list of summary dicts (from RunResult.summary)
    metric    : key in summary dict to plot
    title     : plot title
    out_path  : save figure here if given
    """
    labels = [s["label"] for s in summaries]
    values = [s[metric]  for s in summaries]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color="#3498db", edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
