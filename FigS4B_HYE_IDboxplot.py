#!/usr/bin/env python3
"""
Create grouped boxplots of identifiable protein counts for Human, Yeast,
and E. coli across multiple experimental series.

Each input file represents one series. Protein species are assigned from the
"Protein.Names" column by taking the text after the FIRST underscore and
before the FIRST semicolon. For every sample column, a protein is counted as
identified when its numeric intensity is greater than
IDENTIFICATION_THRESHOLD.

Outputs
-------
1. <prefix>_sample_counts.csv
   Wide table: Series, Sample, Human, Yeast, Ecoli
2. <prefix>_summary_statistics.csv
   Summary statistics for every Series x Species combination
3. <prefix>_boxplot.svg
   Editable-text vector figure
4. <prefix>_boxplot.png
   Raster figure at the configured DPI
5. <prefix>_processing_report.txt
   Processing details and warnings

Required packages
-----------------
pandas, numpy, matplotlib

SciPy is optional. If installed, the 95% confidence interval uses the
Student t distribution; otherwise, the script uses a normal approximation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

try:
    from scipy.stats import t as student_t
except ImportError:  # SciPy is optional
    student_t = None


# =============================================================================
# USER SETTINGS
# =============================================================================

# Detect the directory where this script is saved.
BASE_DIR = Path(__file__).resolve().parent

# ---- Inputs / outputs --------------------------------------------------------
# Each file is one series. The series_name is the label used in the CSV files,
# plot, and legend.
INPUT_FILES: List[Dict[str, str]] = [
    {
        "path": str(BASE_DIR / "Input/FigS4B_LL2106_noMBR_HYE_all_forbox.csv"),
        "series_name": "Extended",
    },
    {
        "path": str(BASE_DIR / "Input/FigS4B_LL2106_noMBR_HYE_4pair_Both_forboxplot.csv"),
        "series_name": "Short-term",
    },
    {
        "path": str(BASE_DIR / "Input/FigS4B_LL2106_Bubis_libfree_mbr_forBoxPlot.csv"),
        "series_name": "Bubis et.al",
    },
]

OUTPUT_DIR: str = str(BASE_DIR / "output/Fig5_HYE")
OUTPUT_PREFIX: str = "HYE_IDbox"

# "csv", "tsv", or "auto". When "auto", the extension determines the format.
FILE_TYPE: str = "csv"
INPUT_ENCODING: str = "utf-8-sig"

# ---- Input-table structure ---------------------------------------------------
PROTEIN_NAME_COLUMN: str = "Protein.Names"

# The first sample/intensity column. "G" means Excel column G (the seventh
# column). An integer is also interpreted as a 1-based Excel column number.
# A longer string can be an exact dataframe column name.
SAMPLE_START_COLUMN: Union[str, int] = "G"

# None means the last column in the file. This accepts the same formats as
# SAMPLE_START_COLUMN.
SAMPLE_END_COLUMN: Optional[Union[str, int]] = None

# If this list is non-empty, it overrides SAMPLE_START_COLUMN and
# SAMPLE_END_COLUMN and uses only these exact column names, in this order.
INCLUDE_SAMPLE_COLUMNS: List[str] = []

# Exact column names to remove after sample-column selection.
EXCLUDE_SAMPLE_COLUMNS: List[str] = [
    # "Average",
    # "CV",
]

# A protein is identified in a sample when intensity > this threshold.
IDENTIFICATION_THRESHOLD: float = 0.0

# ---- Species assignment ------------------------------------------------------
# The species code is extracted from Protein.Names using:
#   text after the FIRST underscore and before the FIRST semicolon.
# Example: TMA7B_HUMAN;TMA7_HUMAN -> HUMAN
SPECIES_CODE_TO_NAME: Dict[str, str] = {
    "HUMAN": "Human",
    "YEAST": "Yeast",
    "ECOLI": "Ecoli",
}

# Order used in the output table and along the x-axis.
SPECIES_ORDER: List[str] = [ "Human", "Yeast", "Ecoli"]#"Human", "Yeast", "Ecoli"

# Display labels can differ from the internal/output names.
SPECIES_DISPLAY_LABELS: Dict[str, str] = {
    "Human": "Human",
    "Yeast": "Yeast",
    "Ecoli": "E.coli",
}

# Set the plotting order of the series. Leave empty to use INPUT_FILES order.
SERIES_ORDER: List[str] = [
    "Extended",
    "Short-term",
    "Bubis et.al",
]

# ---- Box colors and series patterns -----------------------------------------
# "species": box fill color is determined by species, while hatch indicates
#             the series. This is closest to the example bar plot.
# "series":  box fill color is determined by series.
COLOR_BY: str = "species"  # "species" or "series"

SPECIES_COLORS: Dict[str, str] = {
    "Human": "#17769B",
    "Yeast": "#F0642B",
    "Ecoli": "#4CAF2A",
}

# Used only when COLOR_BY = "series". Missing entries receive automatic colors.
SERIES_COLORS: Dict[str, str] = {
    "Series 1": "#4C78A8",
    "Series 2": "#F58518",
    "Series 3": "#54A24B",
}

# Hatches remain useful even when boxes are colored. Missing entries receive
# automatic hatch patterns.
SERIES_HATCHES: Dict[str, str] = {
    "Series 1": "",
    "Series 2": "---",
    "Series 3": "....",
}

# Optional per-series transparency. Missing entries use DEFAULT_BOX_ALPHA.
SERIES_ALPHA: Dict[str, float] = {
    "Series 1": 0.85,
    "Series 2": 0.85,
    "Series 3": 0.85,
}
DEFAULT_BOX_ALPHA: float = 0.85

# ---- Figure dimensions and text ---------------------------------------------
FIGURE_WIDTH: float = 7
FIGURE_HEIGHT: float = 5
FIGURE_DPI: int = 600

FONT_FAMILY: str = "Arial"
FONT_SIZE: float = 10
AXIS_LABEL_SIZE: float = 11
TICK_LABEL_SIZE: float = 10
LEGEND_FONT_SIZE: float = 9

PLOT_TITLE: str = ""
X_AXIS_LABEL: str = ""
Y_AXIS_LABEL: str = "Identifiable proteins per sample"

# ---- Boxplot appearance ------------------------------------------------------
BOX_WIDTH: float = 0.22
GAP_BETWEEN_SERIES: float = 0.045
BOX_LINE_WIDTH: float = 1.2
WHISKER_LINE_WIDTH: float = 1.0
CAP_LINE_WIDTH: float = 1.0
MEDIAN_LINE_WIDTH: float = 1.5

BOX_EDGE_COLOR: str = "black"
WHISKER_COLOR: str = "black"
CAP_COLOR: str = "black"
MEDIAN_COLOR: str = "black"

SHOW_MEAN: bool = True
MEAN_MARKER: str = "D"
MEAN_MARKER_SIZE: float = 5.0
MEAN_MARKER_FACE_COLOR: str = "white"
MEAN_MARKER_EDGE_COLOR: str = "black"

# ---- Optional individual sample points --------------------------------------
SHOW_SAMPLE_POINTS: bool = True
SAMPLE_POINT_SIZE: float = 50.0
SAMPLE_POINT_ALPHA: float = 0.65
SAMPLE_POINT_JITTER: float = 0.055
SAMPLE_POINT_MARKER: str = "o"
SAMPLE_POINT_EDGE_COLOR: str = "black"
SAMPLE_POINT_EDGE_WIDTH: float = 0.35
SAMPLE_POINT_COLOR: str = "match"  # "match" or any matplotlib color
RANDOM_SEED: int = 42

# Boxplot outlier symbols are controlled separately from sample points.
# When SHOW_SAMPLE_POINTS is True, setting this to False avoids drawing
# outliers twice.
SHOW_BOX_OUTLIERS: bool = False
OUTLIER_MARKER: str = "o"
OUTLIER_MARKER_SIZE: float = 4.0
OUTLIER_FACE_COLOR: str = "none"
OUTLIER_EDGE_COLOR: str = "black"

# ---- Axes and legend ---------------------------------------------------------
Y_AXIS_MIN: Optional[float] = 0
Y_AXIS_MAX: Optional[float] = 6000 #None
Y_AXIS_TICK_INTERVAL: Optional[float] = None
USE_THOUSANDS_SEPARATOR: bool = True

SHOW_HORIZONTAL_GRID: bool = False
GRID_ALPHA: float = 0.25
GRID_LINE_STYLE: str = "--"

SHOW_TOP_SPINE: bool = False
SHOW_RIGHT_SPINE: bool = False

SHOW_SERIES_LEGEND: bool = True
SHOW_SPECIES_LEGEND: bool = False
LEGEND_LOCATION: str = "best"
LEGEND_FRAME_ON: bool = False
LEGEND_TITLE: str = "Series"

# ---- Statistics and output formatting ---------------------------------------
QUARTILE_METHOD: str = "linear"
CONFIDENCE_LEVEL: float = 0.95
SUMMARY_DECIMAL_PLACES: int = 4

# SVG text remains editable instead of being converted to vector paths.
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = FONT_FAMILY
matplotlib.rcParams["font.size"] = FONT_SIZE


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def excel_column_to_zero_based(column_letters: str) -> int:
    """Convert Excel column letters (A, B, ..., AA) to a zero-based index."""
    token = column_letters.strip().upper()
    if not token or not token.isalpha():
        raise ValueError(f"Invalid Excel column letters: {column_letters!r}")

    value = 0
    for character in token:
        value = value * 26 + (ord(character) - ord("A") + 1)
    return value - 1


def resolve_column_index(
    reference: Union[str, int],
    columns: Sequence[str],
    setting_name: str,
) -> int:
    """
    Resolve a column reference to a zero-based dataframe index.

    Rules:
    - int: 1-based Excel-style column number
    - short alphabetic string (for example, "G" or "AA"): Excel letters
    - otherwise: exact dataframe column name
    """
    if isinstance(reference, int):
        index = reference - 1
    elif isinstance(reference, str):
        token = reference.strip()
        if token.isalpha() and len(token) <= 3:
            index = excel_column_to_zero_based(token)
        else:
            if token not in columns:
                raise KeyError(
                    f"{setting_name}={reference!r} is not an exact column name "
                    "in the input table."
                )
            index = list(columns).index(token)
    else:
        raise TypeError(
            f"{setting_name} must be an Excel column letter, a 1-based integer, "
            "or an exact column name."
        )

    if index < 0 or index >= len(columns):
        raise IndexError(
            f"{setting_name}={reference!r} resolves to column index {index + 1}, "
            f"but the file contains only {len(columns)} columns."
        )
    return index


def select_sample_columns(columns: Sequence[str]) -> List[str]:
    """Return sample columns according to the user configuration."""
    column_list = list(columns)

    if INCLUDE_SAMPLE_COLUMNS:
        missing = [name for name in INCLUDE_SAMPLE_COLUMNS if name not in column_list]
        if missing:
            raise KeyError(
                "The following INCLUDE_SAMPLE_COLUMNS were not found: "
                + ", ".join(missing)
            )
        selected = list(INCLUDE_SAMPLE_COLUMNS)
    else:
        start_index = resolve_column_index(
            SAMPLE_START_COLUMN, column_list, "SAMPLE_START_COLUMN"
        )
        if SAMPLE_END_COLUMN is None:
            end_index = len(column_list) - 1
        else:
            end_index = resolve_column_index(
                SAMPLE_END_COLUMN, column_list, "SAMPLE_END_COLUMN"
            )

        if end_index < start_index:
            raise ValueError(
                "SAMPLE_END_COLUMN occurs before SAMPLE_START_COLUMN."
            )
        selected = column_list[start_index : end_index + 1]

    selected = [name for name in selected if name not in EXCLUDE_SAMPLE_COLUMNS]

    if not selected:
        raise ValueError("No sample columns remain after column selection.")

    return selected


def determine_separator(path: Path) -> str:
    """Determine the input delimiter from FILE_TYPE or the file extension."""
    file_type = FILE_TYPE.strip().lower()
    if file_type == "csv":
        return ","
    if file_type == "tsv":
        return "\t"
    if file_type == "auto":
        if path.suffix.lower() in {".tsv", ".txt"}:
            return "\t"
        return ","
    raise ValueError('FILE_TYPE must be "csv", "tsv", or "auto".')


def read_input_table(path: Path) -> pd.DataFrame:
    """Read one CSV/TSV file."""
    return pd.read_csv(
        path,
        sep=determine_separator(path),
        encoding=INPUT_ENCODING,
        low_memory=False,
    )


def extract_species_code(protein_name: Any) -> Optional[str]:
    """
    Extract species code after the FIRST underscore and before the FIRST
    semicolon.

    Example:
        TMA7B_HUMAN;TMA7_HUMAN -> HUMAN
    """
    if pd.isna(protein_name):
        return None

    text = str(protein_name).strip()
    if "_" not in text:
        return None

    after_first_underscore = text.split("_", 1)[1]
    species_code = after_first_underscore.split(";", 1)[0].strip().upper()
    return species_code or None


def quantiles(values: np.ndarray) -> Tuple[float, float, float]:
    """Calculate Q1, median, and Q3 with compatibility across NumPy versions."""
    try:
        q1, median, q3 = np.quantile(
            values,
            [0.25, 0.50, 0.75],
            method=QUARTILE_METHOD,
        )
    except TypeError:  # NumPy versions before the method= argument
        q1, median, q3 = np.quantile(
            values,
            [0.25, 0.50, 0.75],
            interpolation=QUARTILE_METHOD,
        )
    return float(q1), float(median), float(q3)


def calculate_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Calculate descriptive statistics and standard 1.5 x IQR whiskers."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        raise ValueError("Cannot calculate statistics for an empty dataset.")

    array = np.sort(array)
    n = int(array.size)
    mean = float(np.mean(array))
    q1, median, q3 = quantiles(array)
    iqr = q3 - q1

    lower_cutoff = q1 - 1.5 * iqr
    upper_cutoff = q3 + 1.5 * iqr

    inlier_values = array[(array >= lower_cutoff) & (array <= upper_cutoff)]
    lower_whisker = float(np.min(inlier_values))
    upper_whisker = float(np.max(inlier_values))

    lower_outliers = array[array < lower_cutoff]
    upper_outliers = array[array > upper_cutoff]
    all_outliers = np.concatenate([lower_outliers, upper_outliers])

    if n > 1:
        standard_deviation = float(np.std(array, ddof=1))
        standard_error = standard_deviation / math.sqrt(n)

        alpha = 1.0 - CONFIDENCE_LEVEL
        if student_t is not None:
            critical_value = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
            ci_method = "Student t"
        else:
            critical_value = 1.959963984540054
            ci_method = "Normal approximation"

        margin = critical_value * standard_error
        ci95_lower = mean - margin
        ci95_upper = mean + margin
    else:
        standard_deviation = np.nan
        standard_error = np.nan
        ci95_lower = np.nan
        ci95_upper = np.nan
        ci_method = "Not calculated (n < 2)"

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_1.5iqr_cutoff": lower_cutoff,
        "upper_1.5iqr_cutoff": upper_cutoff,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "lower_outlier_count": int(lower_outliers.size),
        "upper_outlier_count": int(upper_outliers.size),
        "ci95_lower": float(ci95_lower) if np.isfinite(ci95_lower) else np.nan,
        "ci95_upper": float(ci95_upper) if np.isfinite(ci95_upper) else np.nan,
        "confidence_interval_method": ci_method,
        "fliers": all_outliers.astype(float).tolist(),
    }


def validate_configuration() -> Tuple[List[str], List[str]]:
    """Validate configuration and return resolved series/species order."""
    if not INPUT_FILES:
        raise ValueError("INPUT_FILES is empty.")

    series_names: List[str] = []
    for index, item in enumerate(INPUT_FILES, start=1):
        if not isinstance(item, dict):
            raise TypeError(
                f"INPUT_FILES item {index} must be a dictionary with "
                '"path" and "series_name".'
            )
        if "path" not in item or "series_name" not in item:
            raise KeyError(
                f"INPUT_FILES item {index} must contain both "
                '"path" and "series_name".'
            )
        series_name = str(item["series_name"]).strip()
        if not series_name:
            raise ValueError(f"INPUT_FILES item {index} has an empty series_name.")
        series_names.append(series_name)

    duplicates = sorted({name for name in series_names if series_names.count(name) > 1})
    if duplicates:
        raise ValueError("Series names must be unique: " + ", ".join(duplicates))

    resolved_series_order = list(SERIES_ORDER) if SERIES_ORDER else series_names
    if set(resolved_series_order) != set(series_names) or len(resolved_series_order) != len(series_names):
        raise ValueError(
            "SERIES_ORDER must contain each configured series name exactly once.\n"
            f"Configured series: {series_names}\n"
            f"SERIES_ORDER: {resolved_series_order}"
        )

    valid_species = list(SPECIES_CODE_TO_NAME.values())
    resolved_species_order = list(SPECIES_ORDER)
    if set(resolved_species_order) != set(valid_species) or len(resolved_species_order) != len(valid_species):
        raise ValueError(
            "SPECIES_ORDER must contain each mapped species name exactly once.\n"
            f"Mapped species: {valid_species}\n"
            f"SPECIES_ORDER: {resolved_species_order}"
        )

    color_by = COLOR_BY.strip().lower()
    if color_by not in {"species", "series"}:
        raise ValueError('COLOR_BY must be "species" or "series".')

    if not (0.0 < CONFIDENCE_LEVEL < 1.0):
        raise ValueError("CONFIDENCE_LEVEL must be between 0 and 1.")

    return resolved_series_order, resolved_species_order


def process_one_file(
    file_path: Path,
    series_name: str,
    species_order: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Process one input file and return wide sample counts plus report lines."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    dataframe = read_input_table(file_path)

    if PROTEIN_NAME_COLUMN not in dataframe.columns:
        raise KeyError(
            f'{file_path.name} does not contain required column "{PROTEIN_NAME_COLUMN}".'
        )

    sample_columns = select_sample_columns(dataframe.columns)

    # Convert selected intensity columns to numeric. Blank/non-numeric cells are
    # not counted as identified.
    original_sample_values = dataframe[sample_columns].copy()
    numeric_values = original_sample_values.apply(pd.to_numeric, errors="coerce")

    nonnumeric_mask = original_sample_values.notna() & numeric_values.isna()
    nonnumeric_cell_count = int(nonnumeric_mask.to_numpy().sum())
    missing_numeric_cell_count = int(numeric_values.isna().to_numpy().sum())

    species_codes = dataframe[PROTEIN_NAME_COLUMN].map(extract_species_code)
    species_names = species_codes.map(SPECIES_CODE_TO_NAME)

    recognized_mask = species_names.notna()
    unrecognized_row_count = int((~recognized_mask).sum())

    result = pd.DataFrame(
        {
            "Series": series_name,
            "Sample": sample_columns,
        }
    )

    recognized_counts: Dict[str, int] = {}
    for species in species_order:
        species_mask = species_names.eq(species)
        recognized_counts[species] = int(species_mask.sum())

        # Each protein row is counted once for each sample when intensity > the
        # configured threshold.
        identified = numeric_values.loc[species_mask, :].gt(
            IDENTIFICATION_THRESHOLD
        )
        counts = identified.sum(axis=0).astype(int)
        result[species] = [int(counts[column]) for column in sample_columns]

    report_lines = [
        f"File: {file_path}",
        f"Series: {series_name}",
        f"Rows in input table: {len(dataframe):,}",
        f"Columns in input table: {len(dataframe.columns):,}",
        f"Sample columns selected: {len(sample_columns):,}",
        "Selected sample columns: " + ", ".join(map(str, sample_columns)),
        f"Identification rule: numeric intensity > {IDENTIFICATION_THRESHOLD}",
    ]
    for species in species_order:
        report_lines.append(
            f"Rows assigned to {species}: {recognized_counts[species]:,}"
        )
    report_lines.extend(
        [
            f"Rows with unrecognized/missing species: {unrecognized_row_count:,}",
            f"Non-numeric nonblank cells converted to missing: {nonnumeric_cell_count:,}",
            f"Total missing cells after numeric conversion: {missing_numeric_cell_count:,}",
            "",
        ]
    )

    return result, report_lines


def create_summary_statistics(
    sample_counts: pd.DataFrame,
    series_order: Sequence[str],
    species_order: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    """Create summary table and a lookup used for plotting."""
    summary_rows: List[Dict[str, Any]] = []
    stats_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for series in series_order:
        series_data = sample_counts.loc[sample_counts["Series"].eq(series)]
        if series_data.empty:
            raise ValueError(f"No sample-count rows are available for series {series!r}.")

        for species in species_order:
            values = series_data[species].to_numpy(dtype=float)
            stats = calculate_statistics(values)
            stats_lookup[(series, species)] = stats

            summary_rows.append(
                {
                    "Series": series,
                    "Species": species,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "standard_deviation": stats["standard_deviation"],
                    "standard_error": stats["standard_error"],
                    "minimum": stats["minimum"],
                    "maximum": stats["maximum"],
                    "q1": stats["q1"],
                    "q3": stats["q3"],
                    "iqr": stats["iqr"],
                    "lower_1.5iqr_cutoff": stats["lower_1.5iqr_cutoff"],
                    "upper_1.5iqr_cutoff": stats["upper_1.5iqr_cutoff"],
                    "lower_whisker": stats["lower_whisker"],
                    "upper_whisker": stats["upper_whisker"],
                    "lower_outlier_count": stats["lower_outlier_count"],
                    "upper_outlier_count": stats["upper_outlier_count"],
                    "ci95_lower": stats["ci95_lower"],
                    "ci95_upper": stats["ci95_upper"],
                    "confidence_interval_method": stats[
                        "confidence_interval_method"
                    ],
                }
            )

    summary = pd.DataFrame(summary_rows)
    return summary, stats_lookup


def automatic_color(index: int) -> str:
    """Return a color from Matplotlib's default color cycle."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C78A8"])
    return cycle[index % len(cycle)]


def get_box_color(
    species: str,
    series: str,
    species_index: int,
    series_index: int,
) -> str:
    """Resolve the box color according to COLOR_BY."""
    if COLOR_BY.strip().lower() == "species":
        return SPECIES_COLORS.get(species, automatic_color(species_index))
    return SERIES_COLORS.get(series, automatic_color(series_index))


def get_series_hatch(series: str, series_index: int) -> str:
    """Resolve the hatch pattern for one series."""
    fallback_hatches = ["", "---", "....", "///", "\\\\", "xx", "++"]
    return SERIES_HATCHES.get(series, fallback_hatches[series_index % len(fallback_hatches)])


def create_boxplot(
    sample_counts: pd.DataFrame,
    stats_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    series_order: Sequence[str],
    species_order: Sequence[str],
    svg_path: Path,
    png_path: Path,
) -> None:
    """Create and save the grouped boxplot."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    rng = np.random.default_rng(RANDOM_SEED)

    number_of_series = len(series_order)
    total_group_width = (
        number_of_series * BOX_WIDTH
        + max(0, number_of_series - 1) * GAP_BETWEEN_SERIES
    )
    first_offset = -total_group_width / 2.0 + BOX_WIDTH / 2.0
    series_offsets = [
        first_offset + index * (BOX_WIDTH + GAP_BETWEEN_SERIES)
        for index in range(number_of_series)
    ]

    species_centers = np.arange(len(species_order), dtype=float)

    for species_index, species in enumerate(species_order):
        for series_index, series in enumerate(series_order):
            position = species_centers[species_index] + series_offsets[series_index]
            values = sample_counts.loc[
                sample_counts["Series"].eq(series), species
            ].to_numpy(dtype=float)

            stats = stats_lookup[(series, species)]
            box_color = get_box_color(
                species, series, species_index, series_index
            )
            hatch = get_series_hatch(series, series_index)
            alpha = SERIES_ALPHA.get(series, DEFAULT_BOX_ALPHA)

            bxp_statistics = {
                "label": f"{series} - {species}",
                "med": stats["median"],
                "q1": stats["q1"],
                "q3": stats["q3"],
                "whislo": stats["lower_whisker"],
                "whishi": stats["upper_whisker"],
                "fliers": stats["fliers"],
                "mean": stats["mean"],
            }

            artists = ax.bxp(
                [bxp_statistics],
                positions=[position],
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=SHOW_BOX_OUTLIERS,
                showmeans=SHOW_MEAN,
                manage_ticks=False,
                boxprops={
                    "linewidth": BOX_LINE_WIDTH,
                    "edgecolor": BOX_EDGE_COLOR,
                },
                whiskerprops={
                    "linewidth": WHISKER_LINE_WIDTH,
                    "color": WHISKER_COLOR,
                },
                capprops={
                    "linewidth": CAP_LINE_WIDTH,
                    "color": CAP_COLOR,
                },
                medianprops={
                    "linewidth": MEDIAN_LINE_WIDTH,
                    "color": MEDIAN_COLOR,
                },
                meanprops={
                    "marker": MEAN_MARKER,
                    "markersize": MEAN_MARKER_SIZE,
                    "markerfacecolor": MEAN_MARKER_FACE_COLOR,
                    "markeredgecolor": MEAN_MARKER_EDGE_COLOR,
                    "linestyle": "none",
                },
                flierprops={
                    "marker": OUTLIER_MARKER,
                    "markersize": OUTLIER_MARKER_SIZE,
                    "markerfacecolor": OUTLIER_FACE_COLOR,
                    "markeredgecolor": OUTLIER_EDGE_COLOR,
                    "linestyle": "none",
                },
            )

            box_artist = artists["boxes"][0]
            box_artist.set_facecolor(box_color)
            box_artist.set_alpha(alpha)
            box_artist.set_hatch(hatch)

            if SHOW_SAMPLE_POINTS:
                jitter = rng.uniform(
                    -SAMPLE_POINT_JITTER,
                    SAMPLE_POINT_JITTER,
                    size=len(values),
                )
                point_color = (
                    box_color
                    if SAMPLE_POINT_COLOR.strip().lower() == "match"
                    else SAMPLE_POINT_COLOR
                )
                ax.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=SAMPLE_POINT_SIZE,
                    marker=SAMPLE_POINT_MARKER,
                    facecolors=point_color,
                    edgecolors=SAMPLE_POINT_EDGE_COLOR,
                    linewidths=SAMPLE_POINT_EDGE_WIDTH,
                    alpha=SAMPLE_POINT_ALPHA,
                    zorder=3,
                )

    ax.set_xticks(species_centers)
    ax.set_xticklabels(
        [SPECIES_DISPLAY_LABELS.get(species, species) for species in species_order],
        fontsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
    if PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=AXIS_LABEL_SIZE + 1)

    if Y_AXIS_MIN is not None or Y_AXIS_MAX is not None:
        current_bottom, current_top = ax.get_ylim()
        bottom = current_bottom if Y_AXIS_MIN is None else Y_AXIS_MIN
        top = current_top if Y_AXIS_MAX is None else Y_AXIS_MAX
        ax.set_ylim(bottom=bottom, top=top)

    if Y_AXIS_TICK_INTERVAL is not None:
        ax.yaxis.set_major_locator(MultipleLocator(Y_AXIS_TICK_INTERVAL))

    if USE_THOUSANDS_SEPARATOR:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    if SHOW_HORIZONTAL_GRID:
        ax.yaxis.grid(
            True,
            alpha=GRID_ALPHA,
            linestyle=GRID_LINE_STYLE,
            zorder=0,
        )
        ax.set_axisbelow(True)

    ax.spines["top"].set_visible(SHOW_TOP_SPINE)
    ax.spines["right"].set_visible(SHOW_RIGHT_SPINE)

    # Legends are built manually so series hatches and species colors are clear.
    series_legend = None
    if SHOW_SERIES_LEGEND:
        series_handles = []
        for series_index, series in enumerate(series_order):
            if COLOR_BY.strip().lower() == "series":
                facecolor = SERIES_COLORS.get(series, automatic_color(series_index))
            else:
                facecolor = "white"
            series_handles.append(
                Patch(
                    facecolor=facecolor,
                    edgecolor=BOX_EDGE_COLOR,
                    hatch=get_series_hatch(series, series_index),
                    label=series,
                    alpha=SERIES_ALPHA.get(series, DEFAULT_BOX_ALPHA),
                )
            )
        series_legend = ax.legend(
            handles=series_handles,
            title=LEGEND_TITLE,
            loc=LEGEND_LOCATION,
            frameon=LEGEND_FRAME_ON,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
        )

    if SHOW_SPECIES_LEGEND:
        species_handles = [
            Patch(
                facecolor=SPECIES_COLORS.get(species, automatic_color(index)),
                edgecolor=BOX_EDGE_COLOR,
                label=SPECIES_DISPLAY_LABELS.get(species, species),
                alpha=DEFAULT_BOX_ALPHA,
            )
            for index, species in enumerate(species_order)
        ]
        species_legend = ax.legend(
            handles=species_handles,
            title="Species",
            loc="upper left",
            frameon=LEGEND_FRAME_ON,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
        )
        if series_legend is not None:
            ax.add_artist(series_legend)
        _ = species_legend

    fig.tight_layout()
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_processing_report(
    report_path: Path,
    report_lines: Sequence[str],
    output_paths: Sequence[Path],
) -> None:
    """Write a plain-text processing report."""
    scipy_status = (
        "available; Student t confidence intervals used"
        if student_t is not None
        else "not installed; normal-approximation confidence intervals used"
    )

    header = [
        "HYE identifiable-protein boxplot processing report",
        "=" * 52,
        f"Python: {sys.version.split()[0]}",
        f"pandas: {pd.__version__}",
        f"NumPy: {np.__version__}",
        f"Matplotlib: {matplotlib.__version__}",
        f"SciPy status: {scipy_status}",
        f"Protein-name column: {PROTEIN_NAME_COLUMN}",
        (
            "Species parsing rule: text after the FIRST underscore and before "
            "the FIRST semicolon"
        ),
        "",
    ]

    footer = [
        "Outputs",
        "-------",
        *[str(path) for path in output_paths],
        "",
    ]

    report_path.write_text(
        "\n".join([*header, *report_lines, *footer]),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    series_order, species_order = validate_configuration()

    output_directory = Path(OUTPUT_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)

    all_sample_counts: List[pd.DataFrame] = []
    report_lines: List[str] = []

    for item in INPUT_FILES:
        file_path = Path(item["path"])
        series_name = str(item["series_name"])
        sample_counts, file_report = process_one_file(
            file_path=file_path,
            series_name=series_name,
            species_order=species_order,
        )
        all_sample_counts.append(sample_counts)
        report_lines.extend(file_report)

    combined_counts = pd.concat(all_sample_counts, ignore_index=True)

    # Apply the user-defined series order to both files and plotting.
    combined_counts["Series"] = pd.Categorical(
        combined_counts["Series"],
        categories=series_order,
        ordered=True,
    )
    combined_counts = combined_counts.sort_values(
        ["Series", "Sample"],
        kind="stable",
    ).reset_index(drop=True)
    combined_counts["Series"] = combined_counts["Series"].astype(str)

    # Ensure the requested wide output-column order.
    combined_counts = combined_counts[["Series", "Sample", *species_order]]

    summary, stats_lookup = create_summary_statistics(
        sample_counts=combined_counts,
        series_order=series_order,
        species_order=species_order,
    )

    sample_counts_path = output_directory / f"{OUTPUT_PREFIX}_sample_counts.csv"
    summary_path = output_directory / f"{OUTPUT_PREFIX}_summary_statistics.csv"
    svg_path = output_directory / f"{OUTPUT_PREFIX}_boxplot.svg"
    png_path = output_directory / f"{OUTPUT_PREFIX}_boxplot.png"
    report_path = output_directory / f"{OUTPUT_PREFIX}_processing_report.txt"

    combined_counts.to_csv(sample_counts_path, index=False)

    summary_for_output = summary.copy()
    numeric_columns = summary_for_output.select_dtypes(include=[np.number]).columns
    summary_for_output[numeric_columns] = summary_for_output[numeric_columns].round(
        SUMMARY_DECIMAL_PLACES
    )
    summary_for_output.to_csv(summary_path, index=False, na_rep="")

    create_boxplot(
        sample_counts=combined_counts,
        stats_lookup=stats_lookup,
        series_order=series_order,
        species_order=species_order,
        svg_path=svg_path,
        png_path=png_path,
    )

    output_paths = [
        sample_counts_path,
        summary_path,
        svg_path,
        png_path,
        report_path,
    ]
    write_processing_report(
        report_path=report_path,
        report_lines=report_lines,
        output_paths=output_paths,
    )

    print("Processing complete.")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
