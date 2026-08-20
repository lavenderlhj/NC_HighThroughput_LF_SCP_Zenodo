"""
SCP 5-in-1 Proteomics QC & Visualization (v11: NO long_df, scalable, with timing/progress)
========================================================================================

WHAT'S NEW (per request)
1) Removes long_df entirely (no melt/concat of protein×run rows). Uses wide matrices:
     group_mats[(series, condition)] = proteins x runs intensity matrix
2) Adds stopwatch + progress reporting (tqdm if available, otherwise periodic prints)
3) Better error prints with context + full traceback
4) Stops writing cv_stats_nonpivot_pre_iqr/post_iqr by default (toggle via flag)
5) Makes PRE-IQR OUTPUTS optional (plots/CSVs). Pre-IQR computation is still performed
   internally because post-IQR depends on it (same math/logic preserved).

NOTES ON MATH/LOGIC PRESERVATION
- ID: per-run ID count = number of proteins with intensity > 0
- CV:
  - pre-norm filter: fraction of (missing OR <= intensity_min) <= missing_max_frac
  - median normalization (linear scale): global_median / sample_median
  - per-protein stats computed across runs
  - post-IQR filtering on cv_percent within each (series, condition)
- Volcano:
  - strict QC-aligned universe: post-IQR proteins (overlap between cond_a and cond_b)
  - <= intensity_min treated as missing
  - log2, global-median shift normalization on log2
  - require n_min >= N_MIN_FOR_DE before imputation
  - imputation: MNAR or KNN (same behavior)
  - Welch t-test + BH FDR
- PCA/UMAP: uses same log2 + global-median shift + imputation approach.

"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

# -----------------------------
# Optional deps
# -----------------------------




try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# ================================================================
# 1) Must-Edit CONFIG 
# ================================================================

# Detect the directory where this script is saved
BASE_DIR = Path(__file__).resolve().parent
# ---- Inputs / outputs
INPUT_FILES: List[str] = [
    
  
    str(BASE_DIR /"Input/Fig6EtoH_NC_clusters_Leiden_0727.csv"),  # file 1
    
    
    
]
OUTPUT_DIR: str = str(BASE_DIR /"output/Fig6+7/ByCluster")
OUTPUT_PREFIX: str = "ByLeiden"
FILE_TYPE: str = "csv"  # "tsv" or "csv"

# Condition parsing map (prefix->condition)
CONDITION_MAP: Dict[str, str] = {
      
    
    # "100": "100",
    # "PBS": "PBS",

    "ClusterA": "PBS",
    "ClusterB": "LPS1",
    "ClusterC": "LPS2", 

    # "DM": "DM",



    # Map column prefix to condition name (e.g., "PT_1" -> prefix "PT" -> "PT")
    # the code takes everything before the first underscore as perfix. 
    # "NameOfRun":"ConditionNameOutput"
}

# Series: map series_name to dict with:
SERIES_CONFIG: Dict[str, Dict] = {
    "Leiden": {
        "files": [
                  "Fig6EtoH_NC_clusters_Leiden_0727.csv",
                  # "LL2112_NC_conc.csv",
                  ],   # basename of C:\users\RTK\DN.tsv
        "color": "#1a8dcf",    # example hex color
        "order": 1,
    },

   
    ###  "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"

    #   - "files": list of basenames of INPUT_FILES belonging to that series
    #   - "color": matplotlib color or hex 
    #   - "order": integer for plotting order 
}

# Which modules to run
FLAGS: Dict[str, bool] = {
    "run_id_plots": True,
    #v11.1 still include the runs that are excluded by ID cutoff for the downstream
    # to truly exclude them, deleted those run by hand in the input file
    #so the missing value is looking at all run per condition raw as base
    "run_cv_plots": True,
    "run_venn": True,  # CV modules dependent, support 3 condition ploting, and 3+ condition csv
    "run_volcano": True, # CV modules dependent
    "run_pca": False, # CV modules dependent if on post-IQR, 
    "run_umap": False, # CV modules dependent if on post-IQR, 
}

# Venn setup 
    #support 3 condition ploting, and 3+ condition csv
VENN_SERIES_CONFIG: Dict[str, Dict] = {
    "Leiden": {
        "conditions": ["PBS", "LPS1","LPS2"],
        "colors": {
            "PBS": "#f78c35",  # e.g. purple
            "LPS1": "#268DB6",  # grey
            "LPS2": "#39DBCE",  # grey
        },
    },
    # "SN": {
    #     "conditions": ["PT", "PC", "RT"],
    #     "colors": {
    #         "PT": "#f0dd76",  # yellowish
    #         "PC": "#ff8c00",  # orange
    #         "RT": "#4daf4a",  # green
    #     },
    # },
}

# Volcano CONFIG 
VOLCANO_COMPARISONS: List[Dict[str, Optional[str]]] = [
    # Example: DN series, PT vs PC
    {"series": "Leiden", "cond_a": "LPS1", "cond_b": "PBS"},
    {"series": "Leiden", "cond_a": "LPS2", "cond_b": "PBS"},
    {"series": "Leiden", "cond_a": "LPS1", "cond_b": "LPS2"},
    # {"series": "Cluster", "cond_a": "24hr100", "cond_b": "24hrPBS"},
    # {"series": "10blocks", "cond_a": "48hr100", "cond_b": "48hrPBS"},

    # # Example: SN series, PC vs RC
    # {"series": "SN", "cond_a": "PT", "cond_b": "PC"},
    # {"series": "SN", "cond_a": "PT", "cond_b": "RT"},

    # Example: all series combined, PT vs RT
    # {"series": None, "cond_a": "PT", "cond_b": "RT"},
    # Each item is a dict with:
    #   - "series": name of the series (e.g. "DN", "SN") or None for all series combined
    #   - "cond_a": condition A
    #   - "cond_b": condition B

]

# ================================================================
# 2) Advance CONFIG 
# ================================================================

# ---- Column mapping
COLUMN_CONFIG: Dict[str, str] = {
    "accession": "Protein.Group",     # Column A
    "name": "Protein.Names",               # Column B
    "description": "First.Protein.Description", # Column C
}

INTENSITY_START_COL: int = 6  # First intensity column index (0-based)
                              # If your intensities start at Excel column G, this is 6 (0-based).

# Filtering thresholds
FILTER_CONFIG: Dict[str, float] = {
    "id_run_cutoff": 1,  # for ID plot only; keep proteins with total IDs > this 
                        #(this ID filter currently only apply to ID module not downstream, need to remove )
    "intensity_min": 0.0,  # for CV plot intensity filter (pre-normalization)
    "missing_max_frac": 0.3,  # max allowed missing fraction
    "iqr_multiplier": 1.5,   # IQR multiplier for CV outliers
    "fc_cutoff_log2": 0.58,   # fold-change cutoff on log2 scale (e.g., 1 => 2x, 0.58=1.5x)
    "p_adj_cutoff": 0.05,
}

# Output Options
WRITE_CV_STATS_NONPIVOT: bool = False # default False = not writing *_cv_stats_nonpivot_pre_iqr/post_iqr
CV_EXPORT_PRE_IQR_OUTPUTS: bool = True   # pivot-pre + box-pre + violin-pre (does NOT skip pre-IQR computation)
CV_EXPORT_PRE_NORM_PIVOT: bool = False      # normalized pivot pre-iqr output (does NOT skip pre-IQR computation)

# Volcano CONFIG 
N_MIN_FOR_DE: int = 3 # Minimum number of REAL (non-missing, pre-imputation) runs required per group
                      # for DE testing. Default = 3. Lower to 2 if you only have 2 replicates.

VOLCANO_COLOR_CONFIG: Dict[str, str] = {
    "base": "#BBB8B8",
    "sig_up": "#a01543",
    "sig_down": "#37b8a3",
}
VOLCANO_RANDOM_SEED = 42

# ---- PCA/UMAP CONFIG ----
IMPUTATION_METHOD: str = "MNAR"  # "MNAR" or "KNN"
MNAR_SHIFT: float = 1.8
KNN_N_NEIGHBORS: int = 5
PCA_N_COMPONENTS = 30      # number of PCA dimensions to COMPUTE + EXPORT to CSV
UMAP_N_COMPONENTS = 2      # number of UMAP dimensions to COMPUTE + EXPORT to CSV (usually keep 2)
PCA_UMAP_PROTEIN_SET: str = "both"  #   "post_iqr" = only CV post-IQR quantifiable proteins (recommended QC view)
                                    #   "all"      = all proteins (exploratory view)
                                    #   "both"     = generate both outputs per series
PCA_UMAP_RANDOM_SEED: int = 42     # Reproducibility for MNAR imputation in PCA/UMAP

# ---- PLOTTING STYLE ----
BASE_FONTSIZE: int = 14
SHOW_CV_MEDIAN_CALLOUT: bool = True
mpl.rcParams["svg.fonttype"] = "none"   # keep text as editable in svg
mpl.rcParams["pdf.fonttype"] = 42   # 42 means “embed fonts as TrueType not outlines.” better editable text in PDF
mpl.rcParams["ps.fonttype"] = 42    # better editable text in PostScript



# ================================================================
# 3) Logging / timing / progress
# ================================================================

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def info(msg: str) -> None:
    print(f"[{_ts()}] [INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[{_ts()}] [WARN] {msg}")


def error(msg: str) -> None:
    print(f"[{_ts()}] [ERROR] {msg}", file=sys.stderr)


class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        info(f"==> START: {self.name}")
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc is None:
            info(f"<== END: {self.name} ({dt:.2f}s)")
            return False
        error(f"<== FAIL: {self.name} after {dt:.2f}s")
        error("".join(traceback.format_exception(exc_type, exc, tb)))
        return False  # re-raise


def progress(it: Iterable, desc: str = ""):
    """tqdm if available; otherwise periodic prints."""
    if HAS_TQDM:
        return tqdm(it, desc=desc, leave=True)
    # fallback generator
    try:
        total = len(it)  # type: ignore[arg-type]
    except Exception:
        total = None

    def gen():
        i = 0
        for x in it:
            i += 1
            if total is not None:
                if i in (1, total) or i % 10 == 0:
                    info(f"{desc} ... {i}/{total}")
            else:
                if i == 1 or i % 50 == 0:
                    info(f"{desc} ... {i}")
            yield x
    return gen()


# ================================================================
# 4) Small utilities
# ================================================================

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def style_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": BASE_FONTSIZE + 2,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE - 2,
        "ytick.labelsize": BASE_FONTSIZE - 2,
    })


def save_figure(fig: plt.Figure, base_name: str) -> None:
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{base_name}.png")
    svg_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{base_name}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.5, transparent=True)
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", pad_inches=0.5, transparent=True)
    plt.close(fig)


def load_protein_file(path: str) -> pd.DataFrame:
    if FILE_TYPE.lower() == "tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def get_intensity_columns(df: pd.DataFrame) -> List[str]:
    return df.columns[INTENSITY_START_COL:].tolist()


def parse_condition_from_sample(sample_name: str) -> Optional[str]:
    prefix = sample_name.split("_", 1)[0] if "_" in sample_name else sample_name
    return CONDITION_MAP.get(prefix, None)


def get_series_for_file(file_path: str) -> Optional[str]:
    base = os.path.basename(file_path)
    for s, cfg in SERIES_CONFIG.items():
        if base in cfg.get("files", []):
            return s
    return None


def series_order() -> List[str]:
    keys = list(SERIES_CONFIG.keys())
    if keys and all("order" in SERIES_CONFIG[k] for k in keys):
        return sorted(keys, key=lambda k: SERIES_CONFIG[k]["order"])
    return keys


def condition_order(available: Optional[List[str]] = None) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for _, v in CONDITION_MAP.items():
        if v not in seen:
            ordered.append(v)
            seen.add(v)
    if available is None:
        return ordered
    avail = set(available)
    return [c for c in ordered if c in avail]


def series_palette() -> Dict[str, str]:
    pal: Dict[str, str] = {}
    for s, cfg in SERIES_CONFIG.items():
        col = cfg.get("color", None)
        if col:
            pal[s] = col
    return pal


def tukey_whiskers(vals: np.ndarray, q1: float, q3: float, multiplier: float):
    iqr = q3 - q1
    tuk_low = q1 - multiplier * iqr
    tuk_up = q3 + multiplier * iqr
    vmin = float(np.min(vals)) if len(vals) else np.nan
    vmax = float(np.max(vals)) if len(vals) else np.nan
    whisk_low = float(max(tuk_low, vmin))
    whisk_up = float(min(tuk_up, vmax))
    return tuk_low, tuk_up, whisk_low, whisk_up, vmin, vmax


# ================================================================
# 5) Data model (NO long_df)
# ================================================================

@dataclass
class LoadedFile:
    file_path: str
    series: Optional[str]
    df: pd.DataFrame
    intensity_cols: List[str]


def load_all_files() -> List[LoadedFile]:
    out: List[LoadedFile] = []
    for path in INPUT_FILES:
        base = os.path.basename(path)
        if not os.path.exists(path):
            warn(f"Input file not found: {path}")
            continue
        try:
            df = load_protein_file(path)
        except Exception as e:
            error(f"Failed to read {base}: {e}")
            error(traceback.format_exc())
            continue

        series = get_series_for_file(path)
        if series is None:
            warn(f"File '{base}' not mapped in SERIES_CONFIG -> will be skipped for series-dependent steps.")
        else:
            info(f"File '{base}' mapped to series '{series}'.")

        try:
            intens = get_intensity_columns(df)
        except Exception as e:
            error(f"Failed detecting intensity columns for {base}: {e}")
            error(traceback.format_exc())
            continue

        out.append(LoadedFile(file_path=path, series=series, df=df, intensity_cols=intens))
    return out


def build_group_matrices(files: List[LoadedFile]) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """
    Returns:
      group_mats[(series, condition)] = DataFrame(protein_index x sample columns), raw intensities
      sample_meta: index=sample, cols=[series, condition, file]
    """
    idA, idB, idC = COLUMN_CONFIG["accession"], COLUMN_CONFIG["name"], COLUMN_CONFIG["description"]
    group_mats: Dict[Tuple[str, str], pd.DataFrame] = {}
    meta_rows: List[dict] = []

    for lf in progress(files, desc="Building group matrices"):
        if lf.series is None:
            continue
        df = lf.df
        base = os.path.basename(lf.file_path)

        try:
            prot_index = pd.MultiIndex.from_frame(df[[idA, idB, idC]].copy())
        except Exception as e:
            error(f"[{base}] Failed creating protein MultiIndex: {e}")
            error(traceback.format_exc())
            continue

        block = df[lf.intensity_cols].copy()
        block.index = prot_index
        block = block.apply(pd.to_numeric, errors="coerce")

        for col in block.columns:
            cond = parse_condition_from_sample(col)
            if cond is None:
                continue
            key = (lf.series, cond)
            if key not in group_mats:
                group_mats[key] = pd.DataFrame(index=block.index)

            sample_name = col
            if sample_name in group_mats[key].columns:
                sample_name = f"{base}::{col}"

            group_mats[key][sample_name] = block[col].values
            meta_rows.append({"sample": sample_name, "series": lf.series, "condition": cond, "file": base})

    sample_meta = pd.DataFrame(meta_rows).drop_duplicates().set_index("sample")
    return group_mats, sample_meta


# ================================================================
# 6) ID module
# ================================================================

def compute_run_id_counts(group_mats: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for (series, cond), mat in group_mats.items():
        hit = mat.fillna(0) > 0
        counts = hit.sum(axis=0).astype(int)
        for run, cnt in counts.items():
            rows.append({"run": run, "series": series, "condition": cond, "id_count": int(cnt)})
    return pd.DataFrame(rows)


def summarize_ids(run_counts: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (series, cond), sub in run_counts.groupby(["series", "condition"]):
        vals = sub["id_count"].astype(float).values
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        median = float(np.median(vals))
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        iqr = q3 - q1
        tuk_low = q1 - 1.5 * iqr
        tuk_up = q3 + 1.5 * iqr
        whisk_low = float(max(tuk_low, vmin))
        whisk_up = float(min(tuk_up, vmax))
        outliers = int(((vals < tuk_low) | (vals > tuk_up)).sum())
        rows.append({
            "series": series,
            "condition": cond,
            "mean_id": mean,
            "stdev_id": std,
            "n_runs": n,
            "median_id": median,
            "q1_id": q1,
            "q3_id": q3,
            "abs_min_id": vmin,
            "abs_max_id": vmax,
            "tukey_lower": tuk_low,
            "tukey_upper": tuk_up,
            "lower_whisker": whisk_low,
            "upper_whisker": whisk_up,
            "outlier_count": outliers,
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["series"] = pd.Categorical(summary["series"], categories=series_order(), ordered=True)
        summary["condition"] = pd.Categorical(summary["condition"], categories=condition_order(summary["condition"].unique().tolist()), ordered=True)
        summary = summary.sort_values(["condition", "series"]).reset_index(drop=True)
    return summary


def plot_id_bar(summary: pd.DataFrame, label: str) -> None:
    if summary.empty:
        warn(f"[ID] No data to plot ({label}).")
        return

    ser_order = series_order()
    conds = condition_order(summary["condition"].astype(str).unique().tolist())

    n_series = max(1, len(ser_order))
    x = np.arange(len(conds), dtype=float)
    width = 0.8 / n_series

    fig_w = max(10, 1.8 * len(conds) + 3)
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ymax = np.nanmax((summary["mean_id"] + summary["stdev_id"]).values)
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.15)

    def fmt(v: float) -> str:
        if v is None or np.isnan(v):
            return ""
        return f"{v:,.0f}" if v >= 1000 else f"{v:.0f}"

    pal = series_palette()

    for i, s in enumerate(ser_order):
        sub = summary[summary["series"] == s]
        means, stds = [], []
        for c in conds:
            row = sub[sub["condition"] == c]
            if row.empty:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(float(row["mean_id"].iloc[0]))
                stds.append(float(row["stdev_id"].iloc[0]))

        offsets = x - 0.4 + i * width + width / 2.0
        ax.bar(offsets, means, width, label=s, color=pal.get(s, None))
        ax.errorbar(offsets, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1, capsize=3, capthick=1)

        for bx, m, sd in zip(offsets, means, stds):
            if np.isnan(m):
                continue
            top = m + (sd if np.isfinite(sd) else 0.0)
            off = 0.03 * (ymax if np.isfinite(ymax) and ymax > 0 else max(1.0, top))
            ax.text(bx, top + off, fmt(m), ha="center", va="bottom",
                    fontsize=BASE_FONTSIZE - 4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="none", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Average protein ID per run")
    ax.set_title(f"Protein ID (mean ± SD) per condition ({label})")
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_figure(fig, f"id_bar_{label}")


def run_id(group_mats: Dict[Tuple[str, str], pd.DataFrame]) -> None:
    with StepTimer("ID analysis"):
        run_ids = compute_run_id_counts(group_mats)
        run_ids.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ID_per_run.csv"), index=False)

        summary_raw = summarize_ids(run_ids)
        summary_raw.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ID_summary_by_series_condition.csv"), index=False)
        plot_id_bar(summary_raw, "raw")

        cutoff = float(FILTER_CONFIG.get("id_run_cutoff", 0))
        run_ids_f = run_ids[run_ids["id_count"] > cutoff].copy() if cutoff > 0 else run_ids.copy()
        run_ids_f.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ID_per_run_filtered.csv"), index=False)

        summary_f = summarize_ids(run_ids_f)
        summary_f.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ID_summary_by_series_condition_filtered.csv"), index=False)
        plot_id_bar(summary_f, "filtered")


# ================================================================
# 7) CV module (no long_df)
# ================================================================

def pre_norm_filter_matrix(mat: pd.DataFrame) -> pd.Index:
    thr = float(FILTER_CONFIG["intensity_min"])
    miss_max = float(FILTER_CONFIG["missing_max_frac"])
    low_or_missing = mat.isna() | (mat <= thr)
    frac_low = low_or_missing.mean(axis=1)
    keep = frac_low <= miss_max
    return mat.index[keep]


def median_normalize_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    med_per_sample = mat.median(axis=0, skipna=True)
    global_med = med_per_sample.median(skipna=True)
    safe = med_per_sample.copy()
    safe[(~np.isfinite(safe)) | (safe == 0)] = 1.0
    scale = global_med / safe
    return mat.mul(scale, axis=1)


def compute_per_protein_stats(mat_norm: pd.DataFrame, series: str, cond: str) -> pd.DataFrame:
    idA, idB, idC = COLUMN_CONFIG["accession"], COLUMN_CONFIG["name"], COLUMN_CONFIG["description"]
    rows: List[dict] = []
    mult = float(FILTER_CONFIG["iqr_multiplier"])

    for idx in mat_norm.index:
        vals = mat_norm.loc[idx].dropna().values.astype(float)
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        cv = float(std / mean) if mean != 0 else np.nan
        cvp = cv * 100.0 if np.isfinite(cv) else np.nan

        q1 = float(np.percentile(vals, 25))
        med = float(np.percentile(vals, 50))
        q3 = float(np.percentile(vals, 75))
        tuk_low, tuk_up, whisk_low, whisk_up, vmin, vmax = tukey_whiskers(vals, q1, q3, mult)

        rows.append({
            "series": series,
            "condition": cond,
            idA: idx[0],
            idB: idx[1],
            idC: idx[2],
            "avg_intensity": mean,
            "stdev_intensity": std,
            "cv": cv,
            "cv_percent": cvp,
            "n_samples": n,
            "q1": q1,
            "median": med,
            "q3": q3,
            "min": vmin,
            "max": vmax,
            "tukey_lower": tuk_low,
            "tukey_upper": tuk_up,
            "whisker_lower": whisk_low,
            "whisker_upper": whisk_up,
        })
    return pd.DataFrame(rows)


def cv_distribution_stats(per_protein_stats: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: List[dict] = []
    mult = float(FILTER_CONFIG["iqr_multiplier"])
    for (series, cond), sub in per_protein_stats.groupby(["series", "condition"]):
        cvs = sub["cv_percent"].dropna().values.astype(float)
        if cvs.size == 0:
            continue
        n = int(cvs.size)
        mean = float(np.mean(cvs))
        std = float(np.std(cvs, ddof=1)) if n > 1 else 0.0
        q1 = float(np.percentile(cvs, 25))
        med = float(np.percentile(cvs, 50))
        q3 = float(np.percentile(cvs, 75))
        tuk_low, tuk_up, whisk_low, whisk_up, vmin, vmax = tukey_whiskers(cvs, q1, q3, mult)
        rows.append({
            "series": series,
            "condition": cond,
            "state": label,
            "n_proteins": n,
            "mean_cv_percent": mean,
            "stdev_cv_percent": std,
            "median_cv_percent": med,
            "q1_cv_percent": q1,
            "q3_cv_percent": q3,
            "min_cv_percent": vmin,
            "max_cv_percent": vmax,
            "tukey_lower_cv_percent": tuk_low,
            "tukey_upper_cv_percent": tuk_up,
            "whisker_lower_cv_percent": whisk_low,
            "whisker_upper_cv_percent": whisk_up,
        })
    return pd.DataFrame(rows)


def plot_cv_violins(per_protein_stats: pd.DataFrame, label: str) -> None:
    if per_protein_stats.empty:
        warn(f"[CV] No data to plot ({label}).")
        return

    df = per_protein_stats.copy()
    ser_order = series_order()
    cond_order = condition_order(df["condition"].dropna().unique().tolist())
    pal = series_palette()

    fig_w = max(11, 1.25 * len(cond_order) + 0.4 * len(ser_order) + 4)
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.violinplot(
        data=df, x="condition", y="cv_percent",
        hue="series", order=cond_order,
        hue_order=ser_order if ser_order else None,
        inner="box", cut=0, palette=pal if pal else None,
        ax=ax
    )

   
    if SHOW_CV_MEDIAN_CALLOUT and ser_order and cond_order:
        # local formatting (not in CONFIG)
        callout_fontsize = max(6, BASE_FONTSIZE - 4)
        xoffset_frac = 0.18   # horizontal spacing between series labels within a condition
        ypad_frac = 0.01      # vertical padding above the median (fraction of y-range)

        k = len(ser_order)
        if k > 1:
            offsets = (np.arange(k) - (k - 1) / 2.0) * float(xoffset_frac)
        else:
            offsets = np.array([0.0])

        y0, y1 = ax.get_ylim()
        y_pad = float(ypad_frac) * (y1 - y0)

        for i, cond in enumerate(cond_order):
            for j, ser in enumerate(ser_order):
                sub = df[(df["condition"] == cond) & (df["series"] == ser)]["cv_percent"].dropna()
                if sub.empty:
                    continue
                med = float(np.median(sub.values))
                if not np.isfinite(med):
                    continue

                ax.text(
                    i + offsets[j],
                    med + y_pad,
                    f"{med:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=callout_fontsize,
                    fontweight="bold",
                    bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.5  # 50% transparent white box
                    )  
                )

    ax.set_title(f"Protein CV per condition ({label})")
    ax.set_ylabel("CV (%)")
    ax.set_xlabel("Condition")
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    save_figure(fig, f"cv_violin_{label}_combined")


def run_cv(group_mats: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """
    Returns per_protein_post (post-IQR) for downstream strict universe steps.
    """
    with StepTimer("CV analysis"):
        idA, idB, idC = COLUMN_CONFIG["accession"], COLUMN_CONFIG["name"], COLUMN_CONFIG["description"]

        # (1) prefilter + (2) normalize per (series,condition)
        norm_mats_pre: Dict[Tuple[str, str], pd.DataFrame] = {}
        for (series, cond), mat in progress(list(group_mats.items()), desc="CV prefilter+normalize"):
            keep_idx = pre_norm_filter_matrix(mat)
            mat_f = mat.loc[keep_idx]
            norm_mats_pre[(series, cond)] = median_normalize_matrix(mat_f)

        # Optional: export normalized pivot PRE
        if CV_EXPORT_PRE_NORM_PIVOT:
            with StepTimer("CV export normalized pivot (pre_iqr)"):
                blocks = []
                for (series, cond), m in norm_mats_pre.items():
                    if m.empty:
                        continue
                    m2 = m.copy()
                    m2.columns = [f"{series}|{cond}|{c}" for c in m2.columns]
                    blocks.append(m2)
                if blocks:
                    norm_pivot = pd.concat(blocks, axis=1)
                    norm_pivot.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_filtered_normalized_pivot.csv"))
                else:
                    warn("[CV] No normalized pre_iqr pivot blocks to export.")

        # (3) per-protein stats PRE (computed regardless, used for post-IQR)
        with StepTimer("CV per-protein stats (pre_iqr)"):
            parts = []
            for (series, cond), m in progress(list(norm_mats_pre.items()), desc="CV stats pre_iqr"):
                if m.empty:
                    continue
                parts.append(compute_per_protein_stats(m, series, cond))
            per_pre = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if per_pre.empty:
            warn("[CV] per_protein_pre is empty; cannot proceed to post-IQR.")
            return pd.DataFrame()

        # (4) stop writing nonpivot pre/post by default (requested)
        if WRITE_CV_STATS_NONPIVOT:
            per_pre.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_stats_nonpivot_pre_iqr.csv"), index=False)

        # (5) pre-IQR outputs optional
        if CV_EXPORT_PRE_IQR_OUTPUTS:
            pivot_pre = per_pre.pivot_table(
                index=[idA, idB, idC],
                columns=["series", "condition"],
                values=["avg_intensity", "stdev_intensity", "cv", "cv_percent", "n_samples"],
                aggfunc="first"
            )
            pivot_pre.columns = [f"{s}|{c}|{m}" for (m, s, c) in pivot_pre.columns.to_flat_index()]
            pivot_pre.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_stats_pivot_pre_iqr.csv"))

            box_pre = cv_distribution_stats(per_pre, label="pre_iqr")
            box_pre.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_box_stats_pre_iqr.csv"), index=False)
            plot_cv_violins(per_pre, label="pre_iqr")

        # post-IQR filter by cv_percent within each (series, condition)
        with StepTimer("CV IQR filter (post_iqr)"):
            mult = float(FILTER_CONFIG["iqr_multiplier"])
            keep_frames = []
            for (series, cond), sub in per_pre.groupby(["series", "condition"]):
                cvs = sub["cv_percent"].dropna().values.astype(float)
                if cvs.size == 0:
                    continue
                q1 = np.percentile(cvs, 25)
                q3 = np.percentile(cvs, 75)
                iqr = q3 - q1
                lower = q1 - mult * iqr
                upper = q3 + mult * iqr
                keep_frames.append(sub[(sub["cv_percent"] >= lower) & (sub["cv_percent"] <= upper)])
            per_post = pd.concat(keep_frames, ignore_index=True) if keep_frames else pd.DataFrame()

        if per_post.empty:
            warn("[CV] per_protein_post is empty after IQR filtering.")
            return pd.DataFrame()

        if WRITE_CV_STATS_NONPIVOT:
            per_post.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_stats_nonpivot_post_iqr.csv"), index=False)

        # Export post-IQR stats pivot (this remains, as in v10)
        pivot_post = per_post.pivot_table(
            index=[idA, idB, idC],
            columns=["series", "condition"],
            values=["avg_intensity", "stdev_intensity", "cv", "cv_percent", "n_samples"],
            aggfunc="first"
        )
        pivot_post.columns = [f"{s}|{c}|{m}" for (m, s, c) in pivot_post.columns.to_flat_index()]
        pivot_post.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_stats_pivot_post_iqr.csv"))

        box_post = cv_distribution_stats(per_post, label="post_iqr")
        box_post.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_box_stats_post_iqr.csv"), index=False)
        plot_cv_violins(per_post, label="post_iqr")

        # Export normalized pivot POST
        with StepTimer("CV export normalized pivot (post_iqr)"):
            kept = set(map(tuple, per_post[[idA, idB, idC]].drop_duplicates().values))
            blocks_post = []
            for (series, cond), m in norm_mats_pre.items():
                if m.empty:
                    continue
                idx_keep = [ix for ix in m.index if tuple(ix) in kept]
                m_post = m.loc[idx_keep] if idx_keep else m.iloc[0:0]
                if m_post.empty:
                    continue
                m2 = m_post.copy()
                m2.columns = [f"{series}|{cond}|{c}" for c in m2.columns]
                blocks_post.append(m2)
            if blocks_post:
                norm_pivot_post = pd.concat(blocks_post, axis=1)
                norm_pivot_post.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cv_filtered_normalized_pivot_post_iqr.csv"))
            else:
                warn("[CV] No normalized post_iqr pivot blocks to export.")

        return per_post


# ================================================================
# 8) Volcano (strict universe = post-IQR overlap)
# ================================================================

def impute_missing_log2(matrix_log2: pd.DataFrame) -> pd.DataFrame:
    mat = matrix_log2.copy()

    if IMPUTATION_METHOD.upper() == "MNAR":
        # Column-wise MNAR (same spirit as v10)
        for col in mat.columns:
            col_vals = mat[col]
            obs = col_vals.dropna()
            if obs.empty:
                continue
            mu = float(obs.mean())
            sigma = float(obs.std(ddof=1))
            if (not np.isfinite(sigma)) or sigma <= 0:
                sigma = 1.0
            shift = MNAR_SHIFT * sigma
            n_missing = int(col_vals.isna().sum())
            if n_missing > 0:
                mat.loc[col_vals.isna(), col] = np.random.normal(loc=mu - shift, scale=sigma, size=n_missing)
        return mat

    imputer = KNNImputer(n_neighbors=KNN_N_NEIGHBORS)
    arr = imputer.fit_transform(mat)
    return pd.DataFrame(arr, index=mat.index, columns=mat.columns)


def volcano_for_pair(
    group_mats: Dict[Tuple[str, str], pd.DataFrame],
    detailed_cv_post: pd.DataFrame,
    series: str,
    cond_a: str,
    cond_b: str,
) -> pd.DataFrame:
    idA, idB, idC = COLUMN_CONFIG["accession"], COLUMN_CONFIG["name"], COLUMN_CONFIG["description"]

    mat_a = group_mats.get((series, cond_a), pd.DataFrame())
    mat_b = group_mats.get((series, cond_b), pd.DataFrame())
    if mat_a.empty or mat_b.empty:
        return pd.DataFrame()

    # strict post-IQR overlap universe
    cv_sub = detailed_cv_post[detailed_cv_post["series"] == series]
    cv_a = cv_sub[cv_sub["condition"] == cond_a]
    cv_b = cv_sub[cv_sub["condition"] == cond_b]
    if cv_a.empty or cv_b.empty:
        return pd.DataFrame()

    set_a = set(map(tuple, cv_a[[idA, idB, idC]].drop_duplicates().values))
    set_b = set(map(tuple, cv_b[[idA, idB, idC]].drop_duplicates().values))
    allowed = set_a & set_b

    mat_a = mat_a.loc[mat_a.index.intersection(allowed)]
    mat_b = mat_b.loc[mat_b.index.intersection(allowed)]
    common_idx = mat_a.index.intersection(mat_b.index)
    mat_a = mat_a.loc[common_idx]
    mat_b = mat_b.loc[common_idx]
    if mat_a.empty or mat_b.empty:
        return pd.DataFrame()

    thr = float(FILTER_CONFIG["intensity_min"])
    combined = pd.concat([mat_a, mat_b], axis=1)
    combined = combined.where(combined > thr, np.nan)

    log2_all = np.log2(combined)

    # global-median shift on log2 (same idea as v10)
    sample_meds = log2_all.median(axis=0, skipna=True)
    global_med = sample_meds.median(skipna=True)
    shifts = global_med - sample_meds
    log2_norm = log2_all.add(shifts, axis=1)

    A = log2_norm[mat_a.columns]
    B = log2_norm[mat_b.columns]

    n_a = A.notna().sum(axis=1)
    n_b = B.notna().sum(axis=1)
    n_min = np.minimum(n_a, n_b)

    keep_idx = A.index[n_min >= N_MIN_FOR_DE]
    A = A.loc[keep_idx]
    B = B.loc[keep_idx]
    n_a = n_a.loc[keep_idx]
    n_b = n_b.loc[keep_idx]
    n_min = n_min.loc[keep_idx]

    if A.empty or B.empty:
        return pd.DataFrame()

    has_imp_a = A.isna().any(axis=1)
    has_imp_b = B.isna().any(axis=1)
    has_imputed_any = has_imp_a | has_imp_b


    np.random.seed(VOLCANO_RANDOM_SEED)

    A_imp = impute_missing_log2(A)
    B_imp = impute_missing_log2(B)

    # linear-scale summary stats (match v10-style output)
    A_lin = np.exp2(A_imp)
    B_lin = np.exp2(B_imp)
    avg_intensity_a = A_lin.mean(axis=1)
    avg_intensity_b = B_lin.mean(axis=1)
    stdev_a = A_lin.std(axis=1, ddof=1)
    stdev_b = B_lin.std(axis=1, ddof=1)
    cv_a = stdev_a / avg_intensity_a.replace(0, np.nan)
    cv_b = stdev_b / avg_intensity_b.replace(0, np.nan)

    rows = []
    for idx in A_imp.index:
        va = A_imp.loc[idx].values.astype(float)
        vb = B_imp.loc[idx].values.astype(float)
        tstat, pval = stats.ttest_ind(va, vb, equal_var=False)
        mean_a = float(np.mean(va))
        mean_b = float(np.mean(vb))
        log2_fc = mean_a - mean_b
        rows.append((idx, log2_fc, pval, mean_a, mean_b))

    res = pd.DataFrame(rows, columns=["index", "log2_fc", "p_value", "mean_log2_a", "mean_log2_b"]).set_index("index")

    res["p_adj"] = np.nan
    valid = ~res["p_value"].isna()
    if valid.any():
        pvals = res.loc[valid, "p_value"].values
        _, padj, _, _ = multipletests(pvals, method="fdr_bh")
        res.loc[valid, "p_adj"] = padj

    res["neg_log10_p_adj"] = np.nan
    ok = res["p_adj"] > 0
    res.loc[ok, "neg_log10_p_adj"] = -np.log10(res.loc[ok, "p_adj"])

    fc_cut = float(FILTER_CONFIG["fc_cutoff_log2"])
    p_cut = float(FILTER_CONFIG["p_adj_cutoff"])

    def classify(r) -> str:
        p = r["p_adj"]
        fc = r["log2_fc"]
        if pd.isna(p) or pd.isna(fc):
            return "None"
        if p < p_cut and fc >= fc_cut:
            return "Sig-up"
        if p < p_cut and fc <= -fc_cut:
            return "Sig-down"
        if p >= p_cut and fc >= fc_cut:
            return "Up"
        if p >= p_cut and fc <= -fc_cut:
            return "Down"
        return "None"

    res["significance"] = res.apply(classify, axis=1)
    res["n_a"] = n_a
    res["n_b"] = n_b
    res["n_min"] = n_min
    res["has_imputed_a"] = has_imp_a
    res["has_imputed_b"] = has_imp_b
    res["avg_intensity_a"] = avg_intensity_a.values
    res["avg_intensity_b"] = avg_intensity_b.values
    res["stdev_a"] = stdev_a.values
    res["stdev_b"] = stdev_b.values
    res["cv_a"] = cv_a.values
    res["cv_b"] = cv_b.values
    res["has_imputed_any"] = has_imputed_any.values

    res = res.reset_index()
    res[[idA, idB, idC]] = pd.DataFrame(res["index"].tolist(), index=res.index)
    res.drop(columns=["index"], inplace=True)
    return res


def plot_volcano(res_df: pd.DataFrame, label: str) -> None:
    if res_df.empty:
        return
    base = VOLCANO_COLOR_CONFIG.get("base", "#808080")
    up = VOLCANO_COLOR_CONFIG.get("sig_up", "#e41a1c")
    down = VOLCANO_COLOR_CONFIG.get("sig_down", "#377eb8")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(res_df["log2_fc"], res_df["neg_log10_p_adj"], s=15, alpha=0.6, color=base, label="Other")

    for sig, col in [("Sig-up", up), ("Sig-down", down)]:
        sub = res_df[res_df["significance"] == sig]
        if not sub.empty:
            ax.scatter(sub["log2_fc"], sub["neg_log10_p_adj"], s=25, alpha=0.9, color=col, label=sig)

    fc_cut = float(FILTER_CONFIG["fc_cutoff_log2"])
    ax.axvline(fc_cut, color="black", linestyle="--", linewidth=1)
    ax.axvline(-fc_cut, color="black", linestyle="--", linewidth=1)
    ax.axhline(-np.log10(float(FILTER_CONFIG["p_adj_cutoff"])), color="black", linestyle="--", linewidth=1)

    ax.set_title(f"Volcano: {label}")
    ax.set_xlabel("log2 fold-change (A - B)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"volcano_{label}")


def run_volcano(group_mats: Dict[Tuple[str, str], pd.DataFrame], detailed_cv_post: pd.DataFrame) -> None:
    with StepTimer("Volcano analysis"):
        if detailed_cv_post is None or detailed_cv_post.empty:
            warn("[VOLCANO] detailed_cv_post is empty; skipping.")
            return
        for comp in progress(VOLCANO_COMPARISONS, desc="Volcano comparisons"):
            series = comp.get("series")
            cond_a = comp.get("cond_a")
            cond_b = comp.get("cond_b")
            if not series or not cond_a or not cond_b:
                warn(f"[VOLCANO] bad comp entry: {comp}")
                continue
            label = f"{cond_a}_vs_{cond_b}_series_{series}"
            info(f"[VOLCANO] {label}")
            res = volcano_for_pair(group_mats, detailed_cv_post, series, cond_a, cond_b)
            if res.empty:
                warn(f"[VOLCANO] No results for {label}")
                continue
            res["series"] = series
            res["cond_a"] = cond_a
            res["cond_b"] = cond_b

            ordered = [
                "Protein.Group","Protein.Names","First.Protein.Description","significance",
                "log2_fc","p_value","p_adj","neg_log10_p_adj",
                "mean_log2_a","mean_log2_b","avg_intensity_a","avg_intensity_b","stdev_a","stdev_b","cv_a","cv_b",
                "n_a","n_b","n_min","has_imputed_a","has_imputed_b","has_imputed_any",                
                "series","cond_a","cond_b",
            ]

            # Keep ordered columns first, then append any extras (so you don't accidentally drop new fields)
            res = res[[c for c in ordered if c in res.columns] + [c for c in res.columns if c not in ordered]]


            out_tsv = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_volcano_{cond_a}_vs_{cond_b}_series_{series}.tsv")
            res.to_csv(out_tsv, sep="\t", index=False)
            plot_volcano(res, label)


# ================================================================
# 9) PCA / UMAP (cluster-ready coords; clustering is separate)
# ================================================================

def get_post_iqr_index(detailed_cv_post: pd.DataFrame, series: str) -> set:
    if detailed_cv_post is None or detailed_cv_post.empty:
        return set()
    idA, idB, idC = COLUMN_CONFIG["accession"], COLUMN_CONFIG["name"], COLUMN_CONFIG["description"]
    sub = detailed_cv_post[detailed_cv_post["series"] == series]
    if sub.empty:
        return set()
    return set(map(tuple, sub[[idA, idB, idC]].drop_duplicates().values))


def build_series_matrix(
    group_mats: Dict[Tuple[str, str], pd.DataFrame],
    sample_meta: pd.DataFrame,
    series: str,
    protein_index: Optional[set],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mats = []
    for (s, _cond), m in group_mats.items():
        if s == series and not m.empty:
            mats.append(m)
    if not mats:
        return pd.DataFrame(), pd.DataFrame()

    matrix = pd.concat(mats, axis=1)
    if protein_index is not None:
        keep = [ix for ix in matrix.index if tuple(ix) in protein_index]
        matrix = matrix.loc[keep] if keep else matrix.iloc[0:0]
    if matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    thr = float(FILTER_CONFIG["intensity_min"])
    matrix = matrix.where(matrix > thr, np.nan)
    log2_mat = np.log2(matrix)

    # global median shift on log2
    meds = log2_mat.median(axis=0, skipna=True)
    global_med = meds.median(skipna=True)
    shifts = global_med - meds
    log2_norm = log2_mat.add(shifts, axis=1)

    # impute
    np.random.seed(PCA_UMAP_RANDOM_SEED)
    log2_imp = impute_missing_log2(log2_norm)

    meta = sample_meta.loc[log2_imp.columns, ["series", "condition", "file"]].copy()
    return log2_imp, meta


def run_pca_umap(
    group_mats: Dict[Tuple[str, str], pd.DataFrame],
    sample_meta: pd.DataFrame,
    detailed_cv_post: Optional[pd.DataFrame]
):
    with StepTimer("PCA/UMAP"):
        mode = (PCA_UMAP_PROTEIN_SET or "post_iqr").strip().lower()
        if mode not in {"post_iqr", "all", "both"}:
            warn(f"[PCA/UMAP] Invalid PCA_UMAP_PROTEIN_SET='{PCA_UMAP_PROTEIN_SET}', using 'post_iqr'.")
            mode = "post_iqr"
        modes = ["post_iqr", "all"] if mode == "both" else [mode]

        series_list = sorted({s for (s, _c) in group_mats.keys()})

        n_pcs_cfg = int(PCA_N_COMPONENTS)
        n_umap_cfg = int(UMAP_N_COMPONENTS)

        for s in progress(series_list, desc="PCA/UMAP by series"):
            post_idx = get_post_iqr_index(detailed_cv_post, s) if detailed_cv_post is not None else set()

            for m in modes:
                if m == "post_iqr":
                    if not post_idx:
                        warn(f"[PCA/UMAP] series={s}: no post_iqr proteins, skipping.")
                        continue
                    prot_idx = post_idx
                else:
                    prot_idx = None

                mat_log2_imp, meta = build_series_matrix(group_mats, sample_meta, s, prot_idx)
                if mat_log2_imp.empty:
                    warn(f"[PCA/UMAP] series={s}, proteins={m}: empty matrix.")
                    continue

                X = mat_log2_imp.T.values  # samples x proteins

                coords_out = pd.DataFrame(index=meta.index).join(meta)
                coords_out["protein_set"] = m

                # ----------------------------
                # PCA
                # ----------------------------
                if FLAGS.get("run_pca", False):
                    # n_components can't exceed min(n_samples, n_features)
                    max_pcs = int(min(X.shape[0], X.shape[1]))
                    n_pcs = max(1, min(n_pcs_cfg, max_pcs))
                    if n_pcs < n_pcs_cfg:
                        warn(f"[PCA] Requested {n_pcs_cfg} PCs but max possible is {max_pcs}; using {n_pcs}.")

                    pca = PCA(n_components=n_pcs)
                    pcs = pca.fit_transform(X)  # (samples x n_pcs)
                    exp = pca.explained_variance_ratio_ * 100.0  # length n_pcs

                    # Export PC1..PCn
                    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
                    pcs_df = pd.DataFrame(pcs, columns=pc_cols, index=meta.index)
                    coords_out = coords_out.join(pcs_df)

                    # Export PC1_pct..PCn_pct (same values repeated per row)
                    pc_pct_cols = [f"PC{i}_pct" for i in range(1, n_pcs + 1)]
                    pct_df = pd.DataFrame({pc_pct_cols[i]: exp[i] for i in range(n_pcs)}, index=meta.index)
                    coords_out = coords_out.join(pct_df)

                    # Plot ONLY PC1 vs PC2 with % in axis labels (if available)
                    if n_pcs >= 2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.scatterplot(data=coords_out, x="PC1", y="PC2", hue="condition", s=40, ax=ax)
                        ax.set_title(f"PCA (series={s}, proteins={m})")
                        ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)")
                        ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")
                        ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
                        fig.tight_layout()
                        save_figure(fig, f"pca_samples_{s}_{m}")
                    else:
                        warn(f"[PCA] series={s}, proteins={m}: only 1 PC computed; skipping 2D PCA plot.")

                # ----------------------------
                # UMAP
                # ----------------------------
                if FLAGS.get("run_umap", False):
                    try:
                        import umap  # type: ignore
                    except Exception:
                             warn("[UMAP] umap-learn not installed; skipping UMAP. pip install umap-learn")

                    else:
                        # Compute/export UMAP1..UMAPn, but plot only UMAP1 vs UMAP2
                        n_umap = max(1, int(n_umap_cfg))
                        reducer = umap.UMAP(n_components=n_umap, random_state=PCA_UMAP_RANDOM_SEED)
                        um = reducer.fit_transform(X)  # (samples x n_umap)

                        um_cols = [f"UMAP{i}" for i in range(1, n_umap + 1)]
                        um_df = pd.DataFrame(um, columns=um_cols, index=meta.index)
                        coords_out = coords_out.join(um_df)

                        if n_umap >= 2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.scatterplot(data=coords_out, x="UMAP1", y="UMAP2", hue="condition", s=40, ax=ax)
                            ax.set_title(f"UMAP (series={s}, proteins={m})")
                            ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
                            fig.tight_layout()
                            save_figure(fig, f"umap_samples_{s}_{m}")
                        else:
                            warn(f"[UMAP] series={s}, proteins={m}: only 1 dim computed; skipping 2D UMAP plot.")

                out_csv = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_pca_umap_samples_{s}_{m}.csv")
                coords_out.to_csv(out_csv, index_label="sample")



# ================================================================
# 10) Venn 
# ================================================================

def export_venn_sets_all_series(detailed_cv_post: pd.DataFrame) -> None:
    """
    Writes TWO files:
      1) {OUTPUT_PREFIX}_venn_sets_all_series.csv
         - Venn REGION PARTITIONS (A-only, A&B-only, A&B&C-only, ...)
         - Matches venn plot numbers and v10 behavior.

      2) {OUTPUT_PREFIX}_venn_full_sets_all_series.csv
         - FULL per-condition sets (all proteins in that condition, overlaps included)

    CSV export supports ANY number of conditions.
    Plotting (elsewhere) supports only 2–3.
    """
    if detailed_cv_post is None or detailed_cv_post.empty:
        warn("[VENN] detailed_cv_post empty; cannot export venn sets.")
        return

    if not VENN_SERIES_CONFIG:
        warn("[VENN] No VENN_SERIES_CONFIG specified; cannot export venn sets.")
        return

    idA = COLUMN_CONFIG["accession"]

    df = detailed_cv_post.copy()
    df = df[df["series"].notna() & df["condition"].notna()]
    if df.empty:
        warn("[VENN] No rows with series+condition; cannot export venn sets.")
        return

    region_rows = []
    full_rows = []

    for series_label in sorted(df["series"].unique()):
        cfg = VENN_SERIES_CONFIG.get(series_label)
        if not cfg:
            continue

        conds = cfg.get("conditions", [])
        if len(conds) < 2:
            warn(f"[VENN] Series '{series_label}' has <2 configured conditions; skipping.")
            continue

        df_s = df[df["series"] == series_label]
        if df_s.empty:
            continue

        # condition -> full protein set
        sets = {
            cond: set(
                df_s.loc[df_s["condition"] == cond, idA]
                .dropna()
                .unique()
            )
            for cond in conds
        }

        # ---- FULL SETS (per condition, overlaps included) ----
        for cond in conds:
            s = sets.get(cond, set())
            full_rows.append({
                "series": series_label,
                "condition": cond,
                "n_proteins": len(s),
                "proteins": ";".join(sorted(s)),
            })

        # ---- REGION PARTITIONS (plot-matching) ----
        for r in range(1, len(conds) + 1):
            for combo in combinations(conds, r):
                combo_set = set.intersection(*(sets[c] for c in combo))
                other_conds = [c for c in conds if c not in combo]
                other_union = (
                    set.union(*(sets[c] for c in other_conds))
                    if other_conds else set()
                )
                unique_to_combo = combo_set - other_union

                region_rows.append({
                    "series": series_label,
                    "conditions": "&".join(combo),
                    "n_proteins": len(unique_to_combo),
                    "proteins": ";".join(sorted(unique_to_combo)),
                })

    # ---- write files ----
    out_regions = pd.DataFrame(region_rows)
    out_regions.to_csv(
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_venn_sets_all_series.csv"),
        index=False,
    )

    out_full = pd.DataFrame(full_rows)
    out_full.to_csv(
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_venn_full_sets_all_series.csv"),
        index=False,
    )



def run_venn(detailed_cv_post: pd.DataFrame) -> None:
    if detailed_cv_post is None or detailed_cv_post.empty:
        warn("[VENN] detailed_cv_post empty; skipping.")
        return
    export_venn_sets_all_series(detailed_cv_post)
    try:
        from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles  # type: ignore
    except Exception:
        warn("[VENN] matplotlib-venn not installed; CSV exported, skipping plots. pip install matplotlib-venn")
        return
   

    idA = COLUMN_CONFIG["accession"]
    df = detailed_cv_post.copy()
    series_list = sorted(df["series"].dropna().unique())

    for series in series_list:
        cfg = VENN_SERIES_CONFIG.get(series, None)
        if not cfg:
            continue
        conds = cfg.get("conditions", [])
        colors = cfg.get("colors", {})
        if len(conds) < 2:
            continue

        sets = {c: set(df[(df["series"] == series) & (df["condition"] == c)][idA].dropna().unique()) for c in conds}
        if len(conds) == 2:
            c1, c2 = conds
            fig, ax = plt.subplots(figsize=(6, 6))
            v = venn2([sets[c1], sets[c2]], set_labels=(c1, c2), ax=ax)
            if v is not None:
                if v.get_patch_by_id("10") is not None and colors.get(c1):
                    v.get_patch_by_id("10").set_facecolor(colors[c1]); v.get_patch_by_id("10").set_alpha(0.5)
                if v.get_patch_by_id("01") is not None and colors.get(c2):
                    v.get_patch_by_id("01").set_facecolor(colors[c2]); v.get_patch_by_id("01").set_alpha(0.5)
            circles = venn2_circles([sets[c1], sets[c2]], ax=ax)
            for circle, c in zip(circles, (c1, c2)):
                if colors.get(c):
                    circle.set_edgecolor(colors[c]); circle.set_linewidth(2.0)
            ax.set_title(f"Venn ({series}): {c1} vs {c2}")
            fig.tight_layout()
            save_figure(fig, f"venn_{series}_{c1}_vs_{c2}")

        elif len(conds) == 3:
            c1, c2, c3 = conds
            fig, ax = plt.subplots(figsize=(7, 7))
            v = venn3([sets[c1], sets[c2], sets[c3]], set_labels=(c1, c2, c3), ax=ax)
            if v is not None:
                for c, pid in zip((c1, c2, c3), ("100", "010", "001")):
                    patch = v.get_patch_by_id(pid)
                    if patch is not None and colors.get(c):
                        patch.set_facecolor(colors[c]); patch.set_alpha(0.5)
            circles = venn3_circles([sets[c1], sets[c2], sets[c3]], ax=ax)
            for circle, c in zip(circles, (c1, c2, c3)):
                if colors.get(c):
                    circle.set_edgecolor(colors[c]); circle.set_linewidth(2.0)
            ax.set_title(f"Venn ({series}): {c1}, {c2}, {c3}")
            fig.tight_layout()
            save_figure(fig, f"venn_{series}_{c1}_{c2}_{c3}")
        else:
            warn(f"[VENN] {series}: {len(conds)} conditions configured; only 2/3 are supported for plotting.")


# ================================================================
# 11) MAIN
# ================================================================

def main() -> None:
    style_matplotlib()
    ensure_output_dir()

    with StepTimer("Load files"):
        files = load_all_files()
        if not files:
            error("No valid input files. Check INPUT_FILES.")
            return

    with StepTimer("Build matrices (no long_df)"):
        group_mats, sample_meta = build_group_matrices(files)
        if not group_mats:
            error("No group matrices built. Check SERIES_CONFIG + CONDITION_MAP + INTENSITY_START_COL.")
            return
        info(f"Built {len(group_mats)} (series, condition) matrices.")
        info(f"Total samples tracked: {len(sample_meta)}")

    detailed_cv_post = pd.DataFrame()
    if FLAGS.get("run_id_plots", True):
        run_id(group_mats)

    if FLAGS.get("run_cv_plots", True):
        detailed_cv_post = run_cv(group_mats)

    if FLAGS.get("run_venn", False):
        with StepTimer("Venn"):
            run_venn(detailed_cv_post)

    if FLAGS.get("run_volcano", False):
        run_volcano(group_mats, detailed_cv_post)

    if FLAGS.get("run_pca", False) or FLAGS.get("run_umap", False):
        run_pca_umap(group_mats, sample_meta, detailed_cv_post)

    info("All done.")


if __name__ == "__main__":
    main()
