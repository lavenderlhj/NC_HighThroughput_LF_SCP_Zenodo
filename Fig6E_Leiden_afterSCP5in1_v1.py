#!/usr/bin/env python3
"""
*kNN + Leiden clustering from SCP5in1 PCA output using an editable config block.
*Edit only the USER CONFIGURATION section
*Input requirements:
- one row per cell/sample
- PCA columns named PC1, PC2, PC3, ...
- optional existing UMAP columns, e.g. UMAP1 and UMAP2

The kNN graph and Leiden clustering are built from PCA coordinates. 
Existing UMAP coordinates are used only to display the resulting Leiden labels.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
    import scanpy as sc
except ImportError as exc:
    raise SystemExit(
        "Missing required packages. Install them with:\n"
        "  pip install 'scanpy[leiden]' pandas matplotlib scipy anndata\n"
        "or:\n"
        "  conda install -c conda-forge scanpy python-igraph leidenalg "
        "pandas matplotlib scipy anndata"
    ) from exc


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Detect the directory where this script is saved.
BASE_DIR = Path(__file__).resolve().parent

# ---- Inputs / outputs
INPUT_FILES: List[str] = [

    str(BASE_DIR / "Input/Fig6E_LL2112_0726_Conc_pca_pcs_post_iqr.csv"),
]

OUTPUT_DIR: str = str(BASE_DIR / "output/Fig6+7/Leiden")
OUTPUT_PREFIX: str = "Leiden_v1"
FILE_TYPE: str = "csv"  # "csv" or "tsv"

# If more than one input file is listed, append each input filename stem to the
# output prefix to prevent files from being overwritten.
APPEND_INPUT_STEM_TO_PREFIX: bool = True

# ---- Input column names
SAMPLE_COLUMN: str = "sample"
CONDITION_COLUMN: str = "condition"  # optional; summaries skipped if absent
UMAP_X_COLUMN: str = "UMAP1"         # optional; UMAP plot skipped if absent
UMAP_Y_COLUMN: str = "UMAP2"

# ---- kNN and Leiden parameters
N_PCS: int = 20
N_NEIGHBORS: int = 15
LEIDEN_RESOLUTION: float = 0.30
RANDOM_SEED: int = 42
DISTANCE_METRIC: str = "euclidean"   # euclidean, cosine, correlation, manhattan
LEIDEN_FLAVOR: str = "igraph"        # igraph or leidenalg
LEIDEN_ITERATIONS: int = 2            # -1 = run until convergence
DIRECTED_GRAPH: bool = False          # igraph workflow is forced to False. Undirected graph treats the relatinship as shared. 
SCALE_PCS: bool = False               # usually False for PCA score input

# ---- Cluster labeling
LABEL_ORDER: str = "size"             # "size" or "raw"
CLUSTER_LABEL_PREFIX: str = "Cluster_"

# Optional biological renaming after clustering.
# Keys may be generated labels (Cluster_1) or raw Leiden IDs (0, 1, 2...).
BIOLOGICAL_LABEL_MAP: Dict[str, str] = {
    # "Cluster_1": "LPS-L",
    # "Cluster_2": "PBS",
    # "Cluster_3": "LPS-S",
}

# ---- Plot colors
# Colors are assigned in displayed cluster order. Leave empty for tab20.
CLUSTER_COLORS: List[str] = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
]

# Optional label-specific overrides; these take precedence over CLUSTER_COLORS.
COLOR_OVERRIDES: Dict[str, str] = {
    # "PBS": "#4E79A7",
    # "LPS-L": "#F28E2B",
    # "LPS-S": "#59A14F",
}

# ---- Plot appearance
FIGURE_WIDTH: float = 7.2
FIGURE_HEIGHT: float = 6.0
POINT_SIZE: float = 18.0
POINT_ALPHA: float = 0.80
POINT_MARKER: str = "o"
POINT_EDGE_WIDTH: float = 0.0
PNG_DPI: int = 600
FONT_SIZE: float = 11.0
PLOT_TITLE: str = ""                    # empty = use output prefix
LEGEND_LOCATION: str = "right"          # right, inside, none
SHOW_CLUSTER_SIZE_IN_LEGEND: bool = True
SHOW_GRID: bool = False
EQUAL_ASPECT: bool = False

# ---- Optional intermediate outputs
EXPORT_KNN_EDGE_LIST: bool = True
EXPORT_PC_MATRIX_USED: bool = True

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


# Keep text editable in SVG instead of converting it to vector outlines.
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42


def sorted_pc_columns(df: pd.DataFrame) -> List[str]:
    columns = [str(c) for c in df.columns if re.fullmatch(r"PC\d+", str(c))]
    return sorted(columns, key=lambda c: int(c[2:]))


def read_input_table(path: Path, file_type: str) -> pd.DataFrame:
    file_type = file_type.lower().strip()
    if file_type == "csv":
        return pd.read_csv(path)
    if file_type == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError("FILE_TYPE must be either 'csv' or 'tsv'.")


def make_display_labels(
    raw_labels: pd.Series,
    order_method: str,
    prefix: str,
) -> tuple[pd.Series, Dict[str, str]]:
    raw_as_text = raw_labels.astype(str)

    if order_method == "size":
        order = raw_as_text.value_counts().index.tolist()
    elif order_method == "raw":
        try:
            order = sorted(raw_as_text.unique(), key=lambda x: int(x))
        except ValueError:
            order = sorted(raw_as_text.unique())
    else:
        raise ValueError("LABEL_ORDER must be 'size' or 'raw'.")

    mapping = {raw: f"{prefix}{i + 1}" for i, raw in enumerate(order)}
    return raw_as_text.map(mapping), mapping


def validate_colors(colors: List[str], overrides: Dict[str, str]) -> None:
    for color in colors:
        if not mpl.colors.is_color_like(color):
            raise ValueError(f"Invalid Matplotlib color in CLUSTER_COLORS: {color}")

    for label, color in overrides.items():
        if not mpl.colors.is_color_like(color):
            raise ValueError(f"Invalid color '{color}' for label '{label}'.")


def build_color_mapping(
    labels: List[str],
    colors: List[str],
    overrides: Dict[str, str],
) -> Dict[str, str]:
    default_cycle = plt.get_cmap("tab20")
    result: Dict[str, str] = {}

    for idx, label in enumerate(labels):
        if colors:
            result[label] = colors[idx % len(colors)]
        else:
            result[label] = mpl.colors.to_hex(default_cycle(idx % 20))

    for label, color in overrides.items():
        result[label] = color

    return result


def save_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str,
    output_base: Path,
    title: str,
    color_map: Dict[str, str],
) -> None:
    plot_df = df[[x, y, label_col]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y, label_col])

    if plot_df.empty:
        raise ValueError(f"No valid points available for plotting {x} versus {y}.")

    labels = list(dict.fromkeys(df[label_col].dropna().astype(str).tolist()))
    mpl.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for label in labels:
        sub = plot_df[plot_df[label_col].astype(str) == label]
        legend_label = (
            f"{label} (n={len(sub)})"
            if SHOW_CLUSTER_SIZE_IN_LEGEND
            else label
        )
        ax.scatter(
            sub[x],
            sub[y],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            marker=POINT_MARKER,
            label=legend_label,
            color=color_map[label],
            linewidths=POINT_EDGE_WIDTH,
            edgecolors="black" if POINT_EDGE_WIDTH > 0 else "none",
            rasterized=False,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(SHOW_GRID, alpha=0.20, linewidth=0.6)

    if EQUAL_ASPECT:
        ax.set_aspect("equal", adjustable="datalim")

    if LEGEND_LOCATION == "right":
        ax.legend(
            frameon=False,
            markerscale=1.25,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0,
        )
    elif LEGEND_LOCATION == "inside":
        ax.legend(frameon=False, markerscale=1.25, loc="best")
    elif LEGEND_LOCATION != "none":
        raise ValueError("LEGEND_LOCATION must be 'right', 'inside', or 'none'.")

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def export_sparse_graph_edges(
    distances: sparse.spmatrix,
    connectivities: sparse.spmatrix,
    sample_ids: np.ndarray,
    output_path: Path,
) -> None:
    dist = sparse.coo_matrix(distances)
    conn = sparse.coo_matrix(connectivities)

    dist_df = pd.DataFrame(
        {
            "source_index": dist.row,
            "target_index": dist.col,
            "distance": dist.data,
        }
    )
    conn_df = pd.DataFrame(
        {
            "source_index": conn.row,
            "target_index": conn.col,
            "connectivity": conn.data,
        }
    )

    edges = pd.merge(
        dist_df,
        conn_df,
        on=["source_index", "target_index"],
        how="outer",
    )
    edges["source_sample"] = sample_ids[edges["source_index"].to_numpy(dtype=int)]
    edges["target_sample"] = sample_ids[edges["target_index"].to_numpy(dtype=int)]
    edges = edges[
        [
            "source_index",
            "source_sample",
            "target_index",
            "target_sample",
            "distance",
            "connectivity",
        ]
    ].sort_values(["source_index", "target_index"])
    edges.to_csv(output_path, index=False)


def validate_configuration() -> None:
    if not INPUT_FILES:
        raise ValueError("INPUT_FILES is empty. Add at least one input file.")
    if N_PCS < 2:
        raise ValueError("N_PCS must be at least 2.")
    if N_NEIGHBORS < 2:
        raise ValueError("N_NEIGHBORS must be at least 2.")
    if LEIDEN_RESOLUTION <= 0:
        raise ValueError("LEIDEN_RESOLUTION must be greater than 0.")
    if not 0 <= POINT_ALPHA <= 1:
        raise ValueError("POINT_ALPHA must be between 0 and 1.")
    if POINT_SIZE <= 0 or FIGURE_WIDTH <= 0 or FIGURE_HEIGHT <= 0 or PNG_DPI <= 0:
        raise ValueError("Plot dimensions, point size, and PNG_DPI must be positive.")
    if DISTANCE_METRIC not in {"euclidean", "cosine", "correlation", "manhattan"}:
        raise ValueError("Unsupported DISTANCE_METRIC.")
    if LEIDEN_FLAVOR not in {"igraph", "leidenalg"}:
        raise ValueError("LEIDEN_FLAVOR must be 'igraph' or 'leidenalg'.")
    validate_colors(CLUSTER_COLORS, COLOR_OVERRIDES)


def make_run_prefix(input_path: Path) -> str:
    if len(INPUT_FILES) > 1 and APPEND_INPUT_STEM_TO_PREFIX:
        return f"{OUTPUT_PREFIX}_{input_path.stem}"
    return OUTPUT_PREFIX


def run_one_file(input_path: Path, output_dir: Path, prefix: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = read_input_table(input_path, FILE_TYPE)
    if df.empty:
        raise ValueError(f"Input file contains no rows: {input_path}")

    pc_columns = sorted_pc_columns(df)
    if not pc_columns:
        raise ValueError(
            f"No PC columns found in {input_path.name}. Expected PC1, PC2, PC3, ..."
        )
    if N_PCS > len(pc_columns):
        raise ValueError(
            f"Requested {N_PCS} PCs, but only {len(pc_columns)} were found in "
            f"{input_path.name}."
        )
    if N_NEIGHBORS >= len(df):
        raise ValueError(
            f"N_NEIGHBORS ({N_NEIGHBORS}) must be smaller than the number of "
            f"rows ({len(df)}) in {input_path.name}."
        )

    selected_pcs = pc_columns[:N_PCS]
    pcs = df[selected_pcs].apply(pd.to_numeric, errors="coerce")
    invalid_rows = pcs.index[pcs.isna().any(axis=1)].tolist()
    if invalid_rows:
        raise ValueError(
            f"Missing or nonnumeric PC values found in {len(invalid_rows)} rows of "
            f"{input_path.name}; example row indices: {invalid_rows[:10]}"
        )

    if SAMPLE_COLUMN in df.columns:
        sample_ids = df[SAMPLE_COLUMN].astype(str).to_numpy()
    else:
        sample_ids = np.array([f"row_{i}" for i in range(len(df))], dtype=str)

    pc_matrix = pcs.to_numpy(dtype=np.float32)
    if SCALE_PCS:
        means = pc_matrix.mean(axis=0, keepdims=True)
        stds = pc_matrix.std(axis=0, ddof=0, keepdims=True)
        stds[stds == 0] = 1.0
        pc_matrix = (pc_matrix - means) / stds

    adata = ad.AnnData(X=np.zeros((len(df), 1), dtype=np.float32))
    adata.obs_names = pd.Index(sample_ids).astype(str)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    adata.obsm["X_pca"] = pc_matrix

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_pca",
        metric=DISTANCE_METRIC,
        random_state=RANDOM_SEED,
    )

    leiden_kwargs: Dict[str, Any] = {
        "resolution": LEIDEN_RESOLUTION,
        "random_state": RANDOM_SEED,
        "key_added": "leiden_raw",
        "flavor": LEIDEN_FLAVOR,
        "n_iterations": LEIDEN_ITERATIONS,
        "directed": DIRECTED_GRAPH,
    }
    if LEIDEN_FLAVOR == "igraph":
        leiden_kwargs["directed"] = False

    sc.tl.leiden(adata, **leiden_kwargs)

    out = df.copy()
    out["leiden_raw"] = adata.obs["leiden_raw"].astype(str).to_numpy()
    display_labels, raw_to_display = make_display_labels(
        out["leiden_raw"], LABEL_ORDER, CLUSTER_LABEL_PREFIX
    )
    out["leiden_cluster"] = display_labels

    if BIOLOGICAL_LABEL_MAP:
        out["leiden_label"] = [
            BIOLOGICAL_LABEL_MAP.get(display, BIOLOGICAL_LABEL_MAP.get(raw, display))
            for display, raw in zip(out["leiden_cluster"], out["leiden_raw"])
        ]
    else:
        out["leiden_label"] = out["leiden_cluster"]

    out["leiden_n_pcs"] = N_PCS
    out["leiden_n_neighbors"] = N_NEIGHBORS
    out["leiden_resolution"] = LEIDEN_RESOLUTION
    out["leiden_seed"] = RANDOM_SEED
    out["leiden_metric"] = DISTANCE_METRIC
    out["leiden_flavor"] = LEIDEN_FLAVOR
    out["leiden_iterations"] = LEIDEN_ITERATIONS
    out["leiden_pcs_scaled"] = SCALE_PCS

    out.to_csv(output_dir / f"{prefix}_leiden_labeled_cells.csv", index=False)

    if EXPORT_PC_MATRIX_USED:
        pc_out = pd.DataFrame(pc_matrix, columns=selected_pcs)
        pc_out.insert(0, SAMPLE_COLUMN, sample_ids)
        pc_out.to_csv(output_dir / f"{prefix}_PC_matrix_used.csv", index=False)

    cluster_counts = (
        out.groupby(["leiden_cluster", "leiden_label", "leiden_raw"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    cluster_counts["percent_cells"] = (
        100 * cluster_counts["n_cells"] / len(out)
    ).round(3)
    cluster_counts.to_csv(output_dir / f"{prefix}_cluster_counts.csv", index=False)

    if CONDITION_COLUMN in out.columns:
        count_table = pd.crosstab(out["leiden_label"], out[CONDITION_COLUMN])
        percent_table = (
            pd.crosstab(
                out["leiden_label"],
                out[CONDITION_COLUMN],
                normalize="index",
            )
            * 100
        ).round(3)

        count_table.to_csv(output_dir / f"{prefix}_cluster_condition_counts.csv")
        percent_table.to_csv(output_dir / f"{prefix}_cluster_condition_percent.csv")

        summary = count_table.copy()
        summary.insert(0, "n_cells", summary.sum(axis=1))
        summary.insert(1, "dominant_condition", percent_table.idxmax(axis=1))
        summary.insert(2, "dominant_condition_percent", percent_table.max(axis=1))
        summary.to_csv(output_dir / f"{prefix}_cluster_summary.csv")

    if EXPORT_KNN_EDGE_LIST:
        export_sparse_graph_edges(
            adata.obsp["distances"],
            adata.obsp["connectivities"],
            sample_ids,
            output_dir / f"{prefix}_knn_graph_edges.csv",
        )

    label_map_table = pd.DataFrame(
        {
            "leiden_raw": list(raw_to_display.keys()),
            "leiden_cluster": list(raw_to_display.values()),
        }
    )
    label_map_table["leiden_label"] = label_map_table.apply(
        lambda row: BIOLOGICAL_LABEL_MAP.get(
            str(row["leiden_cluster"]),
            BIOLOGICAL_LABEL_MAP.get(
                str(row["leiden_raw"]),
                str(row["leiden_cluster"]),
            ),
        ),
        axis=1,
    )
    label_map_table.to_csv(
        output_dir / f"{prefix}_cluster_label_map.csv",
        index=False,
    )

    labels_in_order = cluster_counts["leiden_label"].astype(str).tolist()
    color_map = build_color_mapping(
        labels_in_order,
        CLUSTER_COLORS,
        COLOR_OVERRIDES,
    )
    pd.DataFrame(
        {
            "leiden_label": labels_in_order,
            "color": [color_map[label] for label in labels_in_order],
        }
    ).to_csv(output_dir / f"{prefix}_cluster_colors.csv", index=False)

    title_stem = PLOT_TITLE.strip() or prefix
    parameter_text = (
        f"PCs={N_PCS}, neighbors={N_NEIGHBORS}, "
        f"resolution={LEIDEN_RESOLUTION:g}, metric={DISTANCE_METRIC}, "
        f"seed={RANDOM_SEED}"
    )

    save_scatter(
        out,
        "PC1",
        "PC2",
        "leiden_label",
        output_dir / f"{prefix}_PCA_Leiden",
        f"{title_stem}: Leiden clusters on PCA\n{parameter_text}",
        color_map,
    )

    if {UMAP_X_COLUMN, UMAP_Y_COLUMN}.issubset(out.columns):
        save_scatter(
            out,
            UMAP_X_COLUMN,
            UMAP_Y_COLUMN,
            "leiden_label",
            output_dir / f"{prefix}_UMAP_Leiden",
            f"{title_stem}: Leiden clusters on existing UMAP\n{parameter_text}",
            color_map,
        )
    else:
        print(
            f"Note: UMAP plot skipped for {input_path.name} because columns "
            f"'{UMAP_X_COLUMN}' and/or '{UMAP_Y_COLUMN}' were not found.",
            file=sys.stderr,
        )

    manifest = {
        "input_file": str(input_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "output_prefix": prefix,
        "file_type": FILE_TYPE,
        "n_cells": len(df),
        "n_clusters": int(out["leiden_label"].nunique()),
        "pc_columns_used": ",".join(selected_pcs),
        "n_pcs": N_PCS,
        "n_neighbors": N_NEIGHBORS,
        "resolution": LEIDEN_RESOLUTION,
        "seed": RANDOM_SEED,
        "metric": DISTANCE_METRIC,
        "leiden_flavor": LEIDEN_FLAVOR,
        "iterations": LEIDEN_ITERATIONS,
        "directed": False if LEIDEN_FLAVOR == "igraph" else DIRECTED_GRAPH,
        "scale_pcs": SCALE_PCS,
        "sample_column": SAMPLE_COLUMN,
        "condition_column": CONDITION_COLUMN,
        "umap_x": UMAP_X_COLUMN,
        "umap_y": UMAP_Y_COLUMN,
        "cluster_colors": json.dumps(CLUSTER_COLORS),
        "color_overrides": json.dumps(COLOR_OVERRIDES),
        "biological_label_map": json.dumps(BIOLOGICAL_LABEL_MAP),
    }
    pd.DataFrame([manifest]).to_csv(
        output_dir / f"{prefix}_analysis_parameters.csv",
        index=False,
    )
    with (output_dir / f"{prefix}_analysis_parameters.txt").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for key, value in manifest.items():
            handle.write(f"{key}: {value}\n")

    print(f"\nCompleted: {input_path.name}")
    print(f"Cells: {len(out)}")
    print(f"Clusters: {out['leiden_label'].nunique()}")
    print(cluster_counts.to_string(index=False))
    print(f"Output prefix: {prefix}")


def main() -> None:
    validate_configuration()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Script directory: {BASE_DIR}")
    print(f"Output directory: {output_dir.resolve()}")

    for input_file in INPUT_FILES:
        input_path = Path(input_file)
        prefix = make_run_prefix(input_path)
        run_one_file(input_path, output_dir, prefix)

    print(f"\nAll results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
