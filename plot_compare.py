# -*- coding: utf-8 -*-
"""
Aggregate and plot metrics across multiple algorithms and multiple `steps` runs.

For each algorithm we expect multiple CSVs in:
    SFC2/result/<ALG>/episode_summary_<steps>.csv
(and optionally episode_summary_bear.csv — ignored here)

What this script does:
1) Scan all supported algorithms' result folders; for every existing
   `episode_summary_<steps>.csv`, compute the mean over episodes for six metrics.
2) For each metric, produce one summary table (rows = steps, columns = algs)
   and save to: SFC2/result/COMPARE/<metric>_by_steps.csv
3) For each metric, plot a figure with x=steps and y=mean(metric),
   one line per algorithm, saved to: SFC2/result/COMPARE/plots/<metric>_by_steps.png

You can call `compute_means_by_steps_only()` to only generate the tables
without plotting, or call `plot_from_tables_only()` if you already have tables.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Config --------------------
ALG_DIR = {
    "BEAR-SFC":  Path("SFC2/result/BEAR-SFC"),
    "PRANOS":    Path("SFC2/result/PRANOS"),
    "MP-DCBJOH": Path("SFC2/result/MP-DCBJOH"),
    "SBD":       Path("SFC2/result/SBD"),
    "OPTSEP":    Path("SFC2/result/OPTSEP"),
    "DRL":       Path("SFC2/result/DRL"),
    "BEAR":      Path("SFC2/result/BEAR"),
    "BEAR-FULL": Path("SFC2/result/BEAR-FULL"),
}

OUT_DIR_TABLES = Path("SFC2/result/COMPARE")
OUT_DIR_PLOTS  = OUT_DIR_TABLES / "plots"
OUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
OUT_DIR_PLOTS.mkdir(parents=True, exist_ok=True)

METRICS = [
    "avg_latency_ms",
    "place_rate",
    "fo_hit_rate",
    "rel_pred_avg",
    "emp_avail",
    "avg_cost_total",
]

# Mapping from metric name to y-axis label
METRIC_YLABELS = {
    "avg_latency_ms": "SFC Latency (ms)",
    "place_rate": "Placement Success Rate (%)",
    "fo_hit_rate": "Failover Success Rate (%)",
    "rel_pred_avg": "Expected Reliability (%)",
    "emp_avail": "Empirical Availability (%)",
    "avg_cost_total": "Average Cost (units)",
}

# Fixed color and marker maps for algorithms
ALG_COLOR = {
    "BEAR-SFC": "#d62728",  # red
    "PRANOS":   "#1f77b4",
    "MP-DCBJOH": "#2ca02c",
    "SBD":      "#ff7f0e",
    "OPTSEP":   "#9467bd",
    "DRL":      "#17becf",
    "BEAR":     "#8c564b",
    "BEAR-FULL": "#e377c2",
}

ALG_MARKER = {
    "BEAR-SFC": "o",
    "PRANOS":   "s",
    "MP-DCBJOH": "^",
    "SBD":      "v",
    "OPTSEP":   "D",
    "DRL":      "X",
    "BEAR":     "P",
    "BEAR-FULL": "*",
}

LINE_KW = dict(linewidth=2.5, markersize=9, markeredgewidth=0.0)
FIGSIZE = (12, 8)  # 6:4 ratio (3:2)
DPI_SAVE = 300

# If not None, limit to first N episodes per CSV when averaging
LIMIT_EPISODES: int | None = 30

CSV_PATTERN = re.compile(r"episode_summary_(\d+)\.csv$")

ALT_NAMES = {
    "avg_cost_total": ["cost_total_avg", "mean_cost_total"],
    "avg_latency_ms": ["latency_ms_avg", "mean_latency_ms"],
    "emp_avail": ["emp_av", "empirical_availability"],
    "rel_pred_avg": ["rel_pred", "reliability_pred_avg"],
}

# -------------------- Helpers --------------------
def _safe_metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric in df.columns:
        return df[metric]
    for name in ALT_NAMES.get(metric, []):
        if name in df.columns:
            return df[name]
    raise KeyError(f"Metric column '{metric}' not found. Available: {list(df.columns)}")

def _list_step_csvs(dir_path: Path) -> Dict[int, Path]:
    """Return {steps:int -> csv_path} for a result directory."""
    step2path: Dict[int, Path] = {}
    if not dir_path.exists():
        return step2path
    for p in dir_path.glob("episode_summary_*.csv"):
        m = CSV_PATTERN.search(p.name)
        if not m:
            continue
        steps = int(m.group(1))
        step2path[steps] = p
    return dict(sorted(step2path.items(), key=lambda kv: kv[0]))

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure we can slice first N rows by index
    df = df.reset_index(drop=True)
    return df

# -------------------- Core aggregation --------------------
def compute_means_by_steps_only(limit_episodes: int | None = LIMIT_EPISODES,
                                out_dir_tables: Path = OUT_DIR_TABLES) -> Dict[str, Path]:
    """
    Compute mean over episodes for each metric, per algorithm, per steps.
    Save one table per metric at: out_dir_tables/<metric>_by_steps.csv.
    Return a mapping {metric -> csv_path}.
    """
    metric2table: Dict[str, pd.DataFrame] = {}

    # Discover all algorithms' available steps
    alg2steps: Dict[str, Dict[int, Path]] = {
        alg: _list_step_csvs(dir_path) for alg, dir_path in ALG_DIR.items()
    }

    # Union of all steps across algorithms
    all_steps = sorted({s for d in alg2steps.values() for s in d.keys()})
    if not all_steps:
        raise RuntimeError("No episode_summary_<steps>.csv files found.")

    for metric in METRICS:
        rows = []
        for steps in all_steps:
            row = {"steps": steps}
            for alg, step_map in alg2steps.items():
                path = step_map.get(steps)
                if path is None:
                    row[alg] = np.nan
                    continue
                try:
                    df = _read_csv(path)
                    series = _safe_metric_series(df, metric)
                    vals = series.values[:limit_episodes] if (limit_episodes is not None) else series.values
                    row[alg] = float(np.nanmean(vals)) if len(vals) else np.nan
                except Exception as e:
                    print(f"[WARN] {alg} steps={steps} metric={metric}: {e}", file=sys.stderr)
                    row[alg] = np.nan
            rows.append(row)
        tbl = pd.DataFrame(rows).sort_values("steps").reset_index(drop=True)
        out_csv = out_dir_tables / f"{metric}_by_steps.csv"
        tbl.to_csv(out_csv, index=False)
        print(f"[OK] Saved table: {out_csv}")
        metric2table[metric] = out_csv

    return metric2table

# -------------------- Plotting from tables --------------------
def plot_from_tables_only(tables_dir: Path = OUT_DIR_TABLES,
                          plots_dir: Path = OUT_DIR_PLOTS) -> None:
    """Plot curves for each metric using precomputed tables from tables_dir."""
    for metric in METRICS:
        csv_path = tables_dir / f"{metric}_by_steps.csv"
        if not csv_path.exists():
            print(f"[WARN] Table not found for {metric}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if "steps" not in df.columns:
            print(f"[WARN] 'steps' column missing in {csv_path}")
            continue

        x = df["steps"].values
        plt.figure(figsize=FIGSIZE)
        plt.rcParams.update({'font.size': 20})  # doubled font size

        # draw each algorithm's line with fixed color/marker
        for alg in ALG_DIR.keys():
            if alg in df.columns:
                color = ALG_COLOR.get(alg, None)
                marker = ALG_MARKER.get(alg, None)
                plt.plot(
                    x, df[alg].values,
                    label=alg,
                    color=color,
                    marker=marker,
                    **LINE_KW,
                )

        # axes labels and grid
        plt.xlabel("Steps")
        ylabel = METRIC_YLABELS.get(metric, metric)
        plt.ylabel(ylabel)

        # background grid (major & minor)
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

        # legend on top
        ncol = 3 if len(ALG_DIR) <= 6 else 4
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.22),
            ncol=ncol,
            frameon=True,
            fancybox=True,
            edgecolor='black'
        )

        plt.tight_layout()
        out_png = plots_dir / f"{metric}_by_steps.png"
        plt.savefig(out_png, dpi=DPI_SAVE)
        plt.close()
        print(f"[OK] Saved plot: {out_png}")

# -------------------- CLI --------------------
import argparse

def main():
    parser = argparse.ArgumentParser(description="Aggregate and/or plot SFC metrics by steps.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # compute subcommand
    p_comp = sub.add_parser("compute", help="Compute per-steps means tables only.")
    p_comp.add_argument("--limit", type=int, default=LIMIT_EPISODES,
                        help="Max episodes per CSV to average (None for all).")
    p_comp.add_argument("--out", type=Path, default=OUT_DIR_TABLES,
                        help="Output directory for tables.")

    # plot subcommand
    p_plot = sub.add_parser("plot", help="Plot from precomputed tables only.")
    p_plot.add_argument("--tables", type=Path, default=OUT_DIR_TABLES,
                        help="Directory containing <metric>_by_steps.csv files.")
    p_plot.add_argument("--out", type=Path, default=OUT_DIR_PLOTS,
                        help="Output directory for plots.")

    # default: do both (backward compatible)
    parser.add_argument("--both", action="store_true",
                        help="Do both compute and plot using defaults (back-compat).")

    args = parser.parse_args()

    if args.cmd == "compute":
        limit = None if (args.limit is None or str(args.limit).lower() == "none") else int(args.limit)
        tables = compute_means_by_steps_only(limit_episodes=limit, out_dir_tables=args.out)
        # print summary of outputs
        for m, p in tables.items():
            print(f"[OK] {m}: {p}")
        return

    if args.cmd == "plot":
        # Only plot from existing tables
        plot_from_tables_only(tables_dir=args.tables, plots_dir=args.out)
        return

    # default behavior for backward compatibility: compute then plot
    tables = compute_means_by_steps_only(LIMIT_EPISODES, out_dir_tables=OUT_DIR_TABLES)
    plot_from_tables_only(tables_dir=OUT_DIR_TABLES, plots_dir=OUT_DIR_PLOTS)

if __name__ == "__main__":
    main()
