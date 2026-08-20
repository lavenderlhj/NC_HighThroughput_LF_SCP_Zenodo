import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent # Detect the directory where this script is saved

# =========================
# 🔧 USER CONFIG
# =========================

INPUT_PATH = str(BASE_DIR / "Input/Fig4_LL2106_1000SCQC.csv")
OUTPUT_DIR: str = str(BASE_DIR / "output/Fig4_SCQC_Scatter")
OUTPUT_PREFIX = "hela_qc_run_outlier"

X_COL = "run order"

SERIES = ["250pgWet", "SC"]#,"DM"

SERIES_LABELS = {
    "250pgWet": "250pg HeLa",
    "SC": "HeLa Single Cells",
    # "DM": "DM",
}

SERIES_COLORS = {
    "250pgWet": "#2ca02c", 
    "SC": "#1f77b4",
    # "DM": "#ff7f0e",
}

MARKER_SIZE = {
    "250pgWet": 18,
    "SC": 18,
    # "DM": 18,
}

X_LABEL = "Run order"
Y_LABEL = "Protein ID number"
BOX_LABELS = ["QC", "SC"]

# toggles
SHOW_SCATTER_OUTLIERS = False #True to make FigS3, False to make Fig4
SHOW_BOXPLOT_OUTLIERS = False 
HOLLOW_SCATTER_OUTLIERS = True
SHOW_IQR_BAND = True
IQR_BAND_SERIES = ["250pgWet", "SC"]#,"DM"

# =========================
# ⚙️ CORE
# =========================
def compute_box_stats(df):
    rows = []
    for col in SERIES:
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        median = s.median()
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        inside = s[(s >= lower) & (s <= upper)]

        rows.append({
            "series": col,
            "label": SERIES_LABELS[col],
            "n": len(s),
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "lower_1.5iqr_cutoff": lower,
            "upper_1.5iqr_cutoff": upper,
            "lower_whisker": inside.min(),
            "upper_whisker": inside.max(),
            "n_outliers": ((s < lower) | (s > upper)).sum()
        })
    return pd.DataFrame(rows)


def make_plot(df, stats_df):

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "DejaVu Sans"

    stats_map = {r["series"]: r for _, r in stats_df.iterrows()}

    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.4, 1.35], wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1], sharey=ax)

    # IQR band
    if SHOW_IQR_BAND:
        for col in IQR_BAND_SERIES:
            q1 = stats_map[col]["q1"]
            q3 = stats_map[col]["q3"]
            ax.axhspan(q1, q3, color=SERIES_COLORS[col], alpha=0.12)

    # scatter
    for col in SERIES:
        mask = df[col].notna() & df[X_COL].notna()
        x = df.loc[mask, X_COL]
        y = df.loc[mask, col]

        lo = stats_map[col]["lower_1.5iqr_cutoff"]
        hi = stats_map[col]["upper_1.5iqr_cutoff"]

        inside = (y >= lo) & (y <= hi)#dynamic    
        outside = ~inside

        ax.scatter(
            x[inside],
            y[inside],
            s=MARKER_SIZE[col],
            color=SERIES_COLORS[col],
            label=SERIES_LABELS[col],
            linewidths=0,
        )

        if SHOW_SCATTER_OUTLIERS:
            if HOLLOW_SCATTER_OUTLIERS:
                ax.scatter(
                    x[outside],
                    y[outside],
                    s=MARKER_SIZE[col],
                    facecolors="none",
                    edgecolors=SERIES_COLORS[col],
                    linewidths=0.9,
                )
            else:
                ax.scatter(
                    x[outside],
                    y[outside],
                    s=MARKER_SIZE[col],
                    color=SERIES_COLORS[col],
                )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(x=0)#no x padding
    ax.set_ylim(bottom=0,top=5900) # y range

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(SERIES),
        frameon=False,
    )

    # boxplot
    box_data = [df[col].dropna().values for col in SERIES]
    bp = ax_box.boxplot(
        box_data,
        patch_artist=True,
        widths=0.55,
        labels=BOX_LABELS,
        showfliers=SHOW_BOXPLOT_OUTLIERS,
    )
    # color the boxes
    for patch, col in zip(bp["boxes"], SERIES):
        patch.set_facecolor(SERIES_COLORS[col])
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.0)





    ax_box.grid(True, axis="y", alpha=0.25)
    ax_box.set_xlabel("Series")
    ax_box.tick_params(axis="y", left=False, labelleft=False)

    return fig


# =========================
# ▶️ RUN
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
stats_df = compute_box_stats(df)

fig = make_plot(df, stats_df)

# output paths
svg_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}.svg")
png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}.png")
csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_stats.csv")

fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close(fig)

stats_df.to_csv(csv_path, index=False)

print("Saved:")
print(svg_path)
print(png_path)
print(csv_path)