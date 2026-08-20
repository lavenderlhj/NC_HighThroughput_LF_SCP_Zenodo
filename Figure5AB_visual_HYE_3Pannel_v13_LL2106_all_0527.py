#!/usr/bin/env python3
"""
HYE 3-panel plotter - robust, future-proof version

What this script does
---------------------
For each pair/comparison (for example HYEA/HYEB), it generates:

1) A 3-panel figure
   - Top-left:
       forward rolling mean CV of the denominator sample (cv_B)
       plotted against denominator log2 intensity (log2_B),
       plus an exponential fit line.
   - Bottom-left:
       scatter plot of log2FC(A/B) vs denominator log2 intensity.
   - Bottom-right:
       distribution of log2FC(A/B), shown sideways, sharing the same
       Y-axis as the scatter plot.

2) One per-protein CSV per pair
   - <pair>__comparison_data__<mode>.csv
   This is the main output table. It contains:
       protein identity columns
       intensities and CVs
       log2_A, log2_B, log2FC_A_over_B
       outlier flag
       expected value
       mean/median with outliers
       mean/median without outliers
       rolling_mean_cv
       used_in_rolling_cv_plot

3) Combined CSV outputs across all pairs in the run
   - all_pairs__top_exponential_fit__<mode>.csv
   - all_pairs__boxplot_stats__<mode>.csv
   - all_pairs__qc_summary__<mode>.csv

4) Optional combined boxplot
   - all_pairs__boxplot__<mode>.svg / png

Important design choices
------------------------
A) Organism filtering
   An organism is used ONLY if it is defined in BOTH:
     - ORGANISM_COLORS
     - EXPECTED_LOG2FC
   If an organism is missing from either dictionary, it is excluded
   from ALL downstream calculations, plots, and CSV outputs.

B) Forward rolling mean CV
   The top panel uses forward rolling:
     row i uses rows i -> i + ROLLING_WINDOW - 1
   The last (ROLLING_WINDOW - 1) proteins do not have a full forward
   window, so:
     - they stay in comparison_data
     - rolling_mean_cv is NaN for them
     - used_in_rolling_cv_plot = False

C) Robust merge-back of rolling results
   A stable source_row_id is created before the top-panel rolling step.
   rolling_mean_cv is merged back into comparison_data ONLY by source_row_id.
   This avoids accidental duplication that can happen when merging on
   repeated numeric columns.

D) Future-proof pandas behavior
   Outlier flagging avoids groupby.apply, so it does not trigger the
   pandas deprecation warning and is safer for future pandas versions.

Expected input column format
----------------------------
  <pair_name>|<condition>|<metric>
"""

import os
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import gaussian_kde

# Detect the directory where this script is saved
BASE_DIR = Path(__file__).resolve().parent

# ---- Inputs / outputs dynamically resolved using BASE_DIR
INPUT_CSV = str(BASE_DIR / "LL2106_HYE_0527/HYE_LL2106_DNAll_ChPool_input.csv")
OUTPUT_DIR = str(BASE_DIR / "LL2106_HYE_0527/HYERatio")


ORGANISM_COL = "Organism"

IDENTITY_COLUMNS = [
    "Protein.Group",
    "Protein.Names",
    "First.Protein.Description",
]

PAIR_COMPARISONS = {
    "MBRch1": [("HYEA", "HYEB")],
    "MBRch2": [("HYEA", "HYEB")],
    "MBR": [("HYEA", "HYEB")],
    "noMBRch1": [("HYEA", "HYEB")],
    "noMBRch2": [("HYEA", "HYEB")],
    "noMBR": [("HYEA", "HYEB")]
}

ORGANISM_COLORS = {
    "ECOLI": "#54A24B",
    "HUMAN": "#4C78A8",
    "YEAST": "#F58518",
    
}

EXPECTED_LOG2FC = {
    
    "ECOLI": -2.0, 
    "HUMAN": 0.0, 
    "YEAST": 1.0 
}


ROLLING_WINDOW = 100
POINT_SIZE = 18
POINT_ALPHA = 0.45
TOP_POINT_SIZE = 30
TOP_POINT_ALPHA = 0.35
FIGSIZE = (7, 5)
TRANSPARENT_BG = True

DISTRIBUTION_MODE = "kde_default"
KDE_FIXED_BW = 0.25
DENSITY_GRID_N = 500
MAKE_COMBINED_BOXPLOT = True
KEEP_OUTLIERS_FOR_PLOTS = True

X_RANGE_MODE = "manual" # "auto" or "manual"
X_RANGE = (10, 26)
SCATTER_Y_RANGE_MODE = "manual" # "auto" or "manual"
SCATTER_Y_RANGE = (-3, 3)
TOP_Y_RANGE_MODE = "manual" # "auto" or "manual"
TOP_Y_RANGE = (0, 40)

HEADER_PATTERN = re.compile(r"^(?P<pair>[^|]+)\|(?P<condition>[^|]+)\|(?P<metric>.+)$")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_log2(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0, np.nan)
    return np.log2(s)

def parse_columns(columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        m = HEADER_PATTERN.match(col)
        if m:
            rows.append({"original": col, "pair": m.group("pair"), "condition": m.group("condition"), "metric": m.group("metric")})
    return pd.DataFrame(rows)

def get_metric_column(meta_df: pd.DataFrame, pair_name: str, condition: str, metric: str) -> str:
    hit = meta_df[(meta_df["pair"] == pair_name) & (meta_df["condition"] == condition) & (meta_df["metric"] == metric)]
    if hit.empty:
        raise KeyError(f"Missing column for {pair_name} | {condition} | {metric}")
    return hit.iloc[0]["original"]

def dedup_legend(ax, title=None, loc="best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    if seen:
        ax.legend(seen.values(), seen.keys(), frameon=False, title=title, loc=loc)

def mode_suffix(mode: str, fixed_bw: float) -> str:
    if mode == "gaussian":
        return "gaussian"
    if mode == "kde_default":
        return "kde_default"
    if mode == "kde_fixed":
        return f"kde_bw_{str(fixed_bw).replace('.', 'p')}"
    raise ValueError(f"Unsupported DISTRIBUTION_MODE: {mode}")

def box_stats(vals) -> dict:
    vals = pd.Series(vals).dropna().sort_values()
    if len(vals) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan, "lower_whisker": np.nan, "upper_whisker": np.nan, "min": np.nan, "max": np.nan, "n_outliers_iqr": 0}
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    lower_cut = q1 - 1.5 * iqr
    upper_cut = q3 + 1.5 * iqr
    inliers = vals[(vals >= lower_cut) & (vals <= upper_cut)]
    n_outliers = int((vals < lower_cut).sum() + (vals > upper_cut).sum())
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_whisker": float(inliers.min()) if len(inliers) else np.nan,
        "upper_whisker": float(inliers.max()) if len(inliers) else np.nan,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n_outliers_iqr": n_outliers,
    }

def add_outlier_flags(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["is_outlier_iqr"] = False
    if df.empty:
        return df
    for _, idx in df.groupby(["pair", "comparison", "Organism"], dropna=False).groups.items():
        group = df.loc[idx]
        vals = group["log2FC_A_over_B"].dropna()
        if len(vals) < 4:
            continue
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (group["log2FC_A_over_B"] < lower) | (group["log2FC_A_over_B"] > upper)
        df.loc[idx, "is_outlier_iqr"] = mask.fillna(False)
    return df

def gaussian_density(values: np.ndarray, grid: np.ndarray):
    values = pd.Series(values).dropna().astype(float).values
    if len(values) < 2:
        return None
    mu = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    dens = (1.0 / (sd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((grid - mu) / sd) ** 2)
    return {"density": dens, "mean": mu, "sd": sd, "method_label": "Gaussian fit"}

def kde_density(values: np.ndarray, grid: np.ndarray, fixed_bw=None):
    values = pd.Series(values).dropna().astype(float).values
    if len(values) < 2:
        return None
    kde = gaussian_kde(values)
    if fixed_bw is not None:
        kde.covariance_factor = lambda: fixed_bw
        kde._compute_covariance()
    dens = kde(grid)
    mean_val = float(np.mean(values))
    sd_val = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    label = "KDE (default bw)" if fixed_bw is None else f"KDE (bw={fixed_bw})"
    return {"density": dens, "mean": mean_val, "sd": sd_val, "method_label": label}

def compute_distribution(values: np.ndarray, grid: np.ndarray, mode: str, fixed_bw: float):
    if mode == "gaussian":
        return gaussian_density(values, grid)
    if mode == "kde_default":
        return kde_density(values, grid, fixed_bw=None)
    if mode == "kde_fixed":
        return kde_density(values, grid, fixed_bw=fixed_bw)
    raise ValueError(f"Unsupported DISTRIBUTION_MODE: {mode}")

def top_panel_fit(top_df: pd.DataFrame):
    if top_df.empty:
        return None
    top_df = top_df.copy()
    window = ROLLING_WINDOW
    if len(top_df) < window:
        top_df["rolling_mean_cv"] = np.nan
        top_df["used_in_rolling_cv_plot"] = False
        return {"top_df": top_df, "x": np.array([]), "y": np.array([]), "fit_x": None, "fit_y": None, "exp_slope": np.nan, "exp_intercept_log": np.nan, "exp_a": np.nan}
    rolling_cv = top_df["cv_B"][::-1].rolling(window=window, min_periods=window).mean()[::-1]
    top_df["rolling_mean_cv"] = rolling_cv
    top_df["used_in_rolling_cv_plot"] = top_df["rolling_mean_cv"].notna()
    top_df_valid = top_df.loc[top_df["used_in_rolling_cv_plot"]].copy()
    x = top_df_valid["log2_B"].to_numpy()
    y = top_df_valid["rolling_mean_cv"].to_numpy()
    fit_mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if fit_mask.sum() < 10:
        return {"top_df": top_df, "x": x, "y": y, "fit_x": None, "fit_y": None, "exp_slope": np.nan, "exp_intercept_log": np.nan, "exp_a": np.nan}
    x_fit_data = x[fit_mask]
    y_fit_data = y[fit_mask]
    slope, intercept = np.polyfit(x_fit_data, np.log(y_fit_data), 1)
    fit_x = np.linspace(np.nanmin(x_fit_data), np.nanmax(x_fit_data), 400)
    fit_y = np.exp(intercept + slope * fit_x)
    return {"top_df": top_df, "x": x, "y": y, "fit_x": fit_x, "fit_y": fit_y, "exp_slope": float(slope), "exp_intercept_log": float(intercept), "exp_a": float(np.exp(intercept))}

def main():
    ensure_dir(OUTPUT_DIR)

    # ============================================================
    # 1) INPUT PARSING & COLUMN METADATA
    # ============================================================
    print(f"[INFO] Reading input: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    meta_df = parse_columns(df.columns)

    if ORGANISM_COL not in df.columns:
        raise KeyError(f"Organism column '{ORGANISM_COL}' not found in input file.")

    identity_cols_present = [c for c in IDENTITY_COLUMNS if c in df.columns]
    print(f"[INFO] Identity columns found: {identity_cols_present if identity_cols_present else 'none'}")

    # ============================================================
    # 2) GLOBAL ORGANISM FILTER (color + expected ratio required)
    # ============================================================
    allowed_organisms = set(ORGANISM_COLORS.keys()) & set(EXPECTED_LOG2FC.keys())

    missing_ratio = set(ORGANISM_COLORS.keys()) - set(EXPECTED_LOG2FC.keys())
    missing_color = set(EXPECTED_LOG2FC.keys()) - set(ORGANISM_COLORS.keys())

    if missing_ratio:
        print(f"[WARNING] These organisms have colors but no expected ratio and will be excluded: {sorted(missing_ratio)}")
    if missing_color:
        print(f"[WARNING] These organisms have expected ratio but no color and will be excluded: {sorted(missing_color)}")

    if not allowed_organisms:
        raise ValueError("No valid organisms remain. Define matching entries in both ORGANISM_COLORS and EXPECTED_LOG2FC.")

    print(f"[INFO] Organisms allowed globally: {sorted(allowed_organisms)}")

    all_box_rows = []
    all_top_fit_rows = []
    all_qc_rows = []

    # ============================================================
    # 3) LOOP OVER PAIRS
    # ============================================================
    total_pairs = len(PAIR_COMPARISONS)

    for pair_idx, (pair_name, comparisons) in enumerate(PAIR_COMPARISONS.items(), start=1):
        print(f"[INFO] [{pair_idx}/{total_pairs}] Processing pair: {pair_name}")

        fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1, 1.15],
            width_ratios=[1.25, 0.55]
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_dist = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

        pair_comp_rows = []
        x_ranges = []
        scatter_y_ranges = []
        top_y_ranges = []
        plotted_any = False

        # ============================================================
        # 4) LOOP OVER COMPARISONS WITHIN PAIR
        # ============================================================
        for comp_idx, (cond_a, cond_b) in enumerate(comparisons, start=1):
            comp_label = f"{cond_a}/{cond_b}"
            print(f"[INFO]   - comparison [{comp_idx}/{len(comparisons)}]: {pair_name} {comp_label}")

            try:
                intensity_a_col = get_metric_column(meta_df, pair_name, cond_a, "avg_intensity")
                intensity_b_col = get_metric_column(meta_df, pair_name, cond_b, "avg_intensity")
                cv_a_col = get_metric_column(meta_df, pair_name, cond_a, "cv_percent")
                cv_b_col = get_metric_column(meta_df, pair_name, cond_b, "cv_percent")
            except Exception as e:
                print(f"[ERROR]   Missing required columns for {pair_name} {comp_label}: {e}")
                continue

            # ============================================================
            # 5) BUILD WORKING TABLE + GLOBAL ORGANISM FILTER
            # ============================================================
            cols = identity_cols_present + [ORGANISM_COL, intensity_a_col, intensity_b_col, cv_a_col, cv_b_col]
            work = df[cols].copy()

            # Global organism filter: only organisms defined in BOTH dictionaries survive
            work = work[work[ORGANISM_COL].isin(allowed_organisms)].copy()

            if work.empty:
                print(f"[WARNING]   No rows left after organism filter for {pair_name} {comp_label}")
                continue

            work = work.rename(columns={
                intensity_a_col: "intensity_A",
                intensity_b_col: "intensity_B",
                cv_a_col: "cv_A",
                cv_b_col: "cv_B",
                ORGANISM_COL: "Organism",
            })

            work["intensity_A"] = pd.to_numeric(work["intensity_A"], errors="coerce")
            work["intensity_B"] = pd.to_numeric(work["intensity_B"], errors="coerce")
            work["cv_A"] = pd.to_numeric(work["cv_A"], errors="coerce")
            work["cv_B"] = pd.to_numeric(work["cv_B"], errors="coerce")

            # ============================================================
            # 6) COMPUTE LOG2 INTENSITY AND LOG2 FOLD CHANGE
            # ============================================================
            work["log2_A"] = safe_log2(work["intensity_A"])
            work["log2_B"] = safe_log2(work["intensity_B"])

            ratio = work["intensity_A"] / work["intensity_B"]
            ratio = ratio.where((work["intensity_A"] > 0) & (work["intensity_B"] > 0))
            work["log2FC_A_over_B"] = np.log2(ratio)

            # ============================================================
            # 7) BUILD COMPARISON DATASET (shared proteins only)
            # ============================================================
            comp_df = work.dropna(subset=["log2_A", "log2_B", "log2FC_A_over_B"]).copy()

            # Stable source row ID used ONLY for merging rolling results back
            comp_df["source_row_id"] = np.arange(len(comp_df))

            comp_df["pair"] = pair_name
            comp_df["comparison"] = comp_label
            comp_df["expected_log2FC"] = comp_df["Organism"].map(EXPECTED_LOG2FC)

            if comp_df.empty:
                print(f"[WARNING]   No shared proteins after filtering for {pair_name} {comp_label}")
                continue

            # ============================================================
            # 8) OUTLIER DETECTION (IQR-based)
            # ============================================================
            comp_df = add_outlier_flags(comp_df)
            plot_df = comp_df.copy() if KEEP_OUTLIERS_FOR_PLOTS else comp_df.loc[~comp_df["is_outlier_iqr"]].copy()

            # ============================================================
            # 9) COMPUTE PER-ORGANISM SUMMARY (with and without outliers)
            # ============================================================
            summary_all = (
                comp_df.groupby("Organism", dropna=False)["log2FC_A_over_B"]
                .agg(mean_log2FC_all="mean", median_log2FC_all="median")
                .reset_index()
            )

            summary_no_outlier = (
                comp_df.loc[~comp_df["is_outlier_iqr"]]
                .groupby("Organism", dropna=False)["log2FC_A_over_B"]
                .agg(mean_log2FC_no_outlier="mean", median_log2FC_no_outlier="median")
                .reset_index()
            )

            summary = summary_all.merge(
                summary_no_outlier,
                on="Organism",
                how="left",
                validate="one_to_one"
            )

            comp_df = comp_df.merge(summary, on="Organism", how="left", validate="many_to_one")
            plot_df = plot_df.merge(summary, on="Organism", how="left", validate="many_to_one")

            for organism, sub_all in comp_df.groupby("Organism", dropna=False):
                org_name = str(organism)
                expected = EXPECTED_LOG2FC.get(org_name, np.nan)
                sub_plot = plot_df[plot_df["Organism"] == organism]

                n_total = int((work["Organism"] == organism).sum())
                n_shared = len(sub_all)
                n_used = len(sub_plot)
                n_outliers = int(sub_all["is_outlier_iqr"].sum())
                frac_outliers = n_outliers / n_shared if n_shared > 0 else np.nan

                mean_fc = float(sub_all["log2FC_A_over_B"].mean()) if n_shared > 0 else np.nan
                median_fc = float(sub_all["log2FC_A_over_B"].median()) if n_shared > 0 else np.nan
                bias_mean = mean_fc - expected if np.isfinite(expected) else np.nan
                bias_median = median_fc - expected if np.isfinite(expected) else np.nan

                all_qc_rows.append({
                    "pair": pair_name,
                    "comparison": comp_label,
                    "Organism": org_name,
                    "n_total": n_total,
                    "n_shared": n_shared,
                    "n_used_for_plots": n_used,
                    "n_outliers_iqr": n_outliers,
                    "fraction_outliers": frac_outliers,
                    "expected_log2FC": expected,
                    "mean_log2FC": mean_fc,
                    "median_log2FC": median_fc,
                    "bias_mean": bias_mean,
                    "bias_median": bias_median,
                    "abs_bias_mean": abs(bias_mean) if np.isfinite(bias_mean) else np.nan,
                    "keep_outliers_used_for_plots": KEEP_OUTLIERS_FOR_PLOTS,
                })

            # ============================================================
            # 10) TOP PANEL: FORWARD ROLLING CV + EXPONENTIAL FIT
            # ============================================================
            top_df = (
                plot_df.dropna(subset=["log2_B", "cv_B"])
                .sort_values("log2_B")
                .reset_index(drop=True)
                .copy()
            )

            if not top_df.empty:
                top_fit = top_panel_fit(top_df)

                if top_fit is not None:
                    top_df_with_roll = top_fit["top_df"]

                    ax_top.scatter(
                        top_fit["x"],
                        top_fit["y"],
                        s=TOP_POINT_SIZE,
                        alpha=TOP_POINT_ALPHA,
                        color="#B0B0B0",
                        edgecolors="none",
                        label="Rolling mean CV",
                    )

                    if top_fit["fit_x"] is not None:
                        ax_top.plot(
                            top_fit["fit_x"],
                            top_fit["fit_y"],
                            color="black",
                            linewidth=2.2,
                            label="Exponential fit",
                        )

                    valid_top_y = pd.Series(top_fit["y"]).dropna()
                    if not valid_top_y.empty:
                        top_y_ranges.append((valid_top_y.min(), valid_top_y.max()))

                    # ============================================================
                    # 11) MERGE ROLLING CV BACK USING source_row_id (SAFE)
                    # ============================================================
                    top_roll_cols = top_df_with_roll[
                        ["source_row_id", "rolling_mean_cv", "used_in_rolling_cv_plot"]
                    ].copy()

                    comp_df = comp_df.merge(
                        top_roll_cols,
                        on="source_row_id",
                        how="left",
                        validate="one_to_one"
                    )

                    if "used_in_rolling_cv_plot" not in comp_df.columns:
                        comp_df["used_in_rolling_cv_plot"] = False
                    else:
                        comp_df["used_in_rolling_cv_plot"] = comp_df["used_in_rolling_cv_plot"].fillna(False).astype(bool)

                    if "rolling_mean_cv" not in comp_df.columns:
                        comp_df["rolling_mean_cv"] = np.nan

                    all_top_fit_rows.append(
                        pd.DataFrame({
                            "pair": [pair_name],
                            "comparison": [comp_label],
                            "exp_slope": [top_fit["exp_slope"]],
                            "exp_intercept_log": [top_fit["exp_intercept_log"]],
                            "exp_a": [top_fit["exp_a"]],
                            "n_points_used_for_fit": [int(np.isfinite(top_fit["y"]).sum())],
                            "keep_outliers_used_for_plots": [KEEP_OUTLIERS_FOR_PLOTS],
                        })
                    )

            pair_comp_rows.append(comp_df)

            if not plot_df.empty:
                x_ranges.append((plot_df["log2_B"].min(), plot_df["log2_B"].max()))
                scatter_y_ranges.append((plot_df["log2FC_A_over_B"].min(), plot_df["log2FC_A_over_B"].max()))

            # ============================================================
            # 12) SCATTER + DISTRIBUTION PLOTS (PER ORGANISM)
            # ============================================================
            for organism, sub in plot_df.groupby("Organism", dropna=False):
                org_name = str(organism)
                color = ORGANISM_COLORS.get(org_name, None)

                expected_fc = EXPECTED_LOG2FC.get(org_name, np.nan)
                mean_fc = float(sub["log2FC_A_over_B"].mean()) if len(sub) > 0 else np.nan

                ax_scatter.scatter(
                    sub["log2_B"],
                    sub["log2FC_A_over_B"],
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA,
                    color=color,
                    edgecolors="none",
                    label=org_name,
                )

                if np.isfinite(expected_fc):
                    ax_scatter.axhline(expected_fc, color=color, linewidth=1.8, linestyle="--", alpha=0.9)
                if np.isfinite(mean_fc):
                    ax_scatter.axhline(mean_fc, color=color, linewidth=2.2, linestyle="-", alpha=0.95)

                vals = sub["log2FC_A_over_B"].dropna().values
                if len(vals) >= 2:
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))
                    vpad = max((vmax - vmin) * 0.15, 0.5)
                    y_grid = np.linspace(vmin - vpad, vmax + vpad, DENSITY_GRID_N)

                    dist = compute_distribution(vals, y_grid, DISTRIBUTION_MODE, KDE_FIXED_BW)
                    if dist is not None:
                        dens = dist["density"]
                        ax_dist.plot(dens, y_grid, color=color, linewidth=2, label=org_name)
                        ax_dist.fill_betweenx(y_grid, 0, dens, color=color, alpha=0.16)

                        if np.isfinite(expected_fc):
                            ax_dist.axhline(expected_fc, color=color, linewidth=1.8, linestyle="--", alpha=0.9)
                        if np.isfinite(mean_fc):
                            ax_dist.axhline(mean_fc, color=color, linewidth=2.2, linestyle="-", alpha=0.95)

                # ============================================================
                # 13) COLLECT BOXPLOT STATISTICS
                # ============================================================
                base_vals_all = comp_df.loc[comp_df["Organism"] == organism, "log2FC_A_over_B"].dropna()
                base_vals_no_outlier = comp_df.loc[
                    (comp_df["Organism"] == organism) & (~comp_df["is_outlier_iqr"]),
                    "log2FC_A_over_B"
                ].dropna()

                stats = box_stats(base_vals_all)
                expected_ref = EXPECTED_LOG2FC.get(org_name, np.nan)

                mean_all = float(base_vals_all.mean()) if len(base_vals_all) else np.nan
                median_all = float(base_vals_all.median()) if len(base_vals_all) else np.nan

                mean_no_outlier = float(base_vals_no_outlier.mean()) if len(base_vals_no_outlier) else np.nan
                median_no_outlier = float(base_vals_no_outlier.median()) if len(base_vals_no_outlier) else np.nan

                stats.update({
                    "pair": pair_name,
                    "comparison": comp_label,
                    "Organism": org_name,
                    "expected_log2FC": expected_ref,

                    "mean_log2FC_all": mean_all,
                    "median_log2FC_all": median_all,
                    "bias_mean_all": mean_all - expected_ref if np.isfinite(expected_ref) else np.nan,
                    "bias_median_all": median_all - expected_ref if np.isfinite(expected_ref) else np.nan,

                    "mean_log2FC_no_outlier": mean_no_outlier,
                    "median_log2FC_no_outlier": median_no_outlier,
                    "bias_mean_no_outlier": mean_no_outlier - expected_ref if np.isfinite(expected_ref) else np.nan,
                    "bias_median_no_outlier": median_no_outlier - expected_ref if np.isfinite(expected_ref) else np.nan,

                    "keep_outliers_used_for_plots": KEEP_OUTLIERS_FOR_PLOTS,
                    "n_points_total": int((work["Organism"] == organism).sum()),
                    "n_points_shared": int((comp_df["Organism"] == organism).sum()),
                    "n_points_used_for_plots": int((plot_df["Organism"] == organism).sum()),
                })
                all_box_rows.append(stats)

            plotted_any = True

        if not plotted_any:
            print(f"[WARNING] No plottable data for pair {pair_name}")
            plt.close(fig)
            continue

        # ============================================================
        # 14) AXIS RANGE CONTROL (AUTO / MANUAL)
        # ============================================================
        if X_RANGE_MODE == "manual":
            ax_top.set_xlim(*X_RANGE)
            ax_scatter.set_xlim(*X_RANGE)
        elif x_ranges:
            x_min = min(v[0] for v in x_ranges)
            x_max = max(v[1] for v in x_ranges)
            x_pad = max((x_max - x_min) * 0.03, 0.05)
            ax_top.set_xlim(x_min - x_pad, x_max + x_pad)
            ax_scatter.set_xlim(x_min - x_pad, x_max + x_pad)

        if SCATTER_Y_RANGE_MODE == "manual":
            ax_scatter.set_ylim(*SCATTER_Y_RANGE)
            ax_dist.set_ylim(*SCATTER_Y_RANGE)
        elif scatter_y_ranges:
            y_min = min(v[0] for v in scatter_y_ranges)
            y_max = max(v[1] for v in scatter_y_ranges)
            y_pad = max((y_max - y_min) * 0.05, 0.2)
            ax_scatter.set_ylim(y_min - y_pad, y_max + y_pad)
            ax_dist.set_ylim(y_min - y_pad, y_max + y_pad)

        if TOP_Y_RANGE_MODE == "manual":
            ax_top.set_ylim(*TOP_Y_RANGE)
        elif top_y_ranges:
            top_min = min(v[0] for v in top_y_ranges)
            top_max = max(v[1] for v in top_y_ranges)
            top_pad = max((top_max - top_min) * 0.10, 0.5)
            ax_top.set_ylim(max(0, top_min - top_pad), top_max + top_pad)

        ax_top.set_title(f"{pair_name}: rolling mean CV vs denominator log2 intensity")
        ax_top.set_xlabel("Denominator log2 intensity")
        ax_top.set_ylabel("Rolling mean denominator CV (%)")

        ax_scatter.set_title(f"{pair_name}: log2FC vs denominator log2 intensity")
        ax_scatter.set_xlabel("Denominator log2 intensity")
        ax_scatter.set_ylabel("log2FC numerator/denominator")
        ax_scatter.axhline(0, linestyle=":", linewidth=1, color="gray")

        dist_title = "Gaussian fit" if DISTRIBUTION_MODE == "gaussian" else ("KDE (default bw)" if DISTRIBUTION_MODE == "kde_default" else f"KDE (bw={KDE_FIXED_BW})")
        ax_dist.set_title(f"{pair_name}: {dist_title}")
        ax_dist.set_xlabel("Density")
        ax_dist.axhline(0, linestyle=":", linewidth=1, color="gray")
        plt.setp(ax_dist.get_yticklabels(), visible=False)
        ax_dist.tick_params(axis="y", length=0)

        dedup_legend(ax_top, loc="best")
        dedup_legend(ax_scatter, title="Organism", loc="best")
        dedup_legend(ax_dist, title="Organism", loc="best")

        suffix = mode_suffix(DISTRIBUTION_MODE, KDE_FIXED_BW)
        svg_path = os.path.join(OUTPUT_DIR, f"{pair_name}__3panel__{suffix}.svg")
        png_path = os.path.join(OUTPUT_DIR, f"{pair_name}__3panel__{suffix}.png")

        # ============================================================
        # 15) SAVE FIGURE OUTPUT
        # ============================================================
        fig.savefig(svg_path, transparent=TRANSPARENT_BG)
        fig.savefig(png_path, dpi=180, transparent=TRANSPARENT_BG)
        plt.close(fig)
        print(f"[INFO]   Saved figure: {svg_path}")

        # ============================================================
        # 16) SAVE PER-PAIR COMPARISON CSV
        # ============================================================
        if pair_comp_rows:
            pd.concat(pair_comp_rows, ignore_index=True).to_csv(
                os.path.join(OUTPUT_DIR, f"{pair_name}__comparison_data__{suffix}.csv"),
                index=False
            )

    suffix = mode_suffix(DISTRIBUTION_MODE, KDE_FIXED_BW)

    # ============================================================
    # 17) SAVE COMBINED OUTPUTS (ALL PAIRS)
    # ============================================================
    if all_top_fit_rows:
        pd.concat(all_top_fit_rows, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, f"all_pairs__top_exponential_fit__{suffix}.csv"),
            index=False
        )
        print("[INFO] Saved combined top-fit CSV")

    if all_box_rows:
        pd.DataFrame(all_box_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"all_pairs__boxplot_stats__{suffix}.csv"),
            index=False
        )
        print("[INFO] Saved combined boxplot stats CSV")

    if all_qc_rows:
        pd.DataFrame(all_qc_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"all_pairs__qc_summary__{suffix}.csv"),
            index=False
        )
        print("[INFO] Saved QC summary CSV")

    # ============================================================
    # 18) BUILD COMBINED BOXPLOT (ALL PAIRS)
    # ============================================================
    if MAKE_COMBINED_BOXPLOT and all_box_rows:
        print("[INFO] Building combined boxplot figure")
        comp_files = [
            os.path.join(OUTPUT_DIR, f"{pair_name}__comparison_data__{suffix}.csv")
            for pair_name in PAIR_COMPARISONS.keys()
            if os.path.exists(os.path.join(OUTPUT_DIR, f"{pair_name}__comparison_data__{suffix}.csv"))
        ]

        if comp_files:
            raw_fc_df = pd.concat([pd.read_csv(f) for f in comp_files], ignore_index=True)

            if not KEEP_OUTLIERS_FOR_PLOTS and "is_outlier_iqr" in raw_fc_df.columns:
                raw_fc_df = raw_fc_df.loc[~raw_fc_df["is_outlier_iqr"]].copy()

            raw_fc_df = raw_fc_df.loc[raw_fc_df["Organism"].isin(allowed_organisms)].copy()

            if raw_fc_df.empty:
                print("[WARNING] No data left for combined boxplot after filtering.")
            else:
                organisms = [org for org in ORGANISM_COLORS.keys() if org in raw_fc_df["Organism"].dropna().unique()]
                pairs = list(PAIR_COMPARISONS.keys())

                fig_width = max(6, 1.2 * len(organisms) + 0.7 * len(pairs))
                fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

                base_positions = np.arange(len(organisms))

                # Dynamic box width depends on number of pairs; outlier dots are hidden
                n_pairs = max(len(pairs), 1)
                cluster_span = 0.8
                width = min(0.28, cluster_span / n_pairs)

                if n_pairs == 1:
                    offsets = np.array([0.0])
                else:
                    offsets = np.linspace(
                        -cluster_span / 2 + width / 2,
                        cluster_span / 2 - width / 2,
                        n_pairs
                    )

                for i, pair_name in enumerate(pairs):
                    color = plt.cm.tab20(i % 20)

                    for j, org in enumerate(organisms):
                        vals = raw_fc_df.loc[
                            (raw_fc_df["pair"] == pair_name)
                            & (raw_fc_df["Organism"] == org),
                            "log2FC_A_over_B"
                        ].dropna().values

                        if len(vals) == 0:
                            continue

                        pos = base_positions[j] + offsets[i]
                        bp = ax.boxplot(
                            vals,
                            positions=[pos],
                            widths=width * 0.9,
                            patch_artist=True,
                            manage_ticks=False,
                            showfliers=False
                        )

                        for patch in bp["boxes"]:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.60)
                            patch.set_linewidth(1.2)

                        for med in bp["medians"]:
                            med.set_color("black")
                            med.set_linewidth(1.2)

                        for whisker in bp["whiskers"]:
                            whisker.set_linewidth(1.1)

                        for cap in bp["caps"]:
                            cap.set_linewidth(1.1)

                legend_handles = [
                    Patch(facecolor=plt.cm.tab20(i % 20), alpha=0.6, label=p)
                    for i, p in enumerate(pairs)
                ]
                ax.legend(
                    handles=legend_handles,
                    frameon=False,
                    title="Pair",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left"
                )

                ax.set_xticks(base_positions)
                ax.set_xticklabels(organisms)
                ax.set_xlim(-0.5, len(organisms) - 0.5)
                ax.set_ylabel("log2FC")
                ax.set_title("Fold-change boxplot by organism, with pair as series")
                ax.axhline(0, linestyle=":", linewidth=1, color="gray")

                fig.savefig(os.path.join(OUTPUT_DIR, f"all_pairs__boxplot__{suffix}.svg"), transparent=TRANSPARENT_BG)
                fig.savefig(os.path.join(OUTPUT_DIR, f"all_pairs__boxplot__{suffix}.png"), dpi=180, transparent=TRANSPARENT_BG)
                plt.close(fig)
                print("[INFO] Saved combined boxplot figure")

    print("[INFO] Done()")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        raise
