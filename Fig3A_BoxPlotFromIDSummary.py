from pathlib import Path
from typing import List
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams["svg.fonttype"] = "none" # Editable SVG text
BASE_DIR = Path(__file__).resolve().parent # Detect the directory where this script is saved

# ---- Inputs / outputs


# ============================================================
# User settings
# ============================================================

# Columns expected in your summary input table:
# condition,n_runs,median_id,q1_id,q3_id,lower_whisker,upper_whisker,quantifiable
SUMMARY_FILE = str(BASE_DIR / "Input/Fig3_WetDry_ID_forBoxPlot_0528.csv")
OUTPUT_DIR: str = str(BASE_DIR / "output/Fig3_WetDry_Boxplot")




BOX_COLORS = {
    "WetCh1": "#7889c0",
    "WetCh2":"#7889c0",
    "Wet": "#7889c0",
    "DryCh1": "#f4c5ad",
    "DryCh2": "#f4c5ad",
    "Dry": "#f4c5ad",
}

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(SUMMARY_FILE)

required_cols = [
    "condition",
    "n_runs",
    "median_id",
    "q1_id",
    "q3_id",
    "lower_whisker",
    "upper_whisker",
    "quantifiable", #n_proteins from cv_box_stats_post_iqr.csv
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in summary file: {missing}")

# ============================================================
# Build matplotlib boxplot stats
# ============================================================
box_stats = []

for _, row in df.iterrows():
    box_stats.append(
        dict(
            label=row["condition"],
            n=int(row["n_runs"]),
            med=float(row["median_id"]),
            q1=float(row["q1_id"]),
            q3=float(row["q3_id"]),
            whislo=float(row["lower_whisker"]),
            whishi=float(row["upper_whisker"]),
            fliers=[],
        )
    )

quantifiable = df["quantifiable"].astype(float).tolist()

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(4.8, 6.4), dpi=300)

# Transparent background
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

box = ax.bxp(
    box_stats,
    showfliers=False,
    patch_artist=True,
    widths=0.68,
    medianprops=dict(color="#222222", linewidth=2.2),
    whiskerprops=dict(color="#222222", linewidth=2.2),
    capprops=dict(color="#222222", linewidth=2.2),
    boxprops=dict(edgecolor="#222222", linewidth=2.2),
)

# Box colors
for patch, stat in zip(box["boxes"], box_stats):
    patch.set_facecolor(BOX_COLORS.get(stat["label"], "#cccccc"))

# Median labels inside boxes
for i, stat in enumerate(box_stats, start=1):
    ax.text(
        i,
        stat["med"] + 60,
        f"{int(round(stat['med']))}",
        ha="center",
        va="bottom",
        fontsize=18,
        color="black",
    )

# n labels above boxes
for i, stat in enumerate(box_stats, start=1):
    ax.text(
        i,
        stat["q3"] + 180,
        f"n={stat['n']}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#222222",
    )

# Quantifiable protein diamond markers
for i, q in enumerate(quantifiable, start=1):
    ax.scatter(
        i + 0.22,
        q,
        marker="D",
        s=90,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.5,
        zorder=5,
        label="Quantifiable proteins" if i == 1 else None,
    )

# Legend
ax.legend(
    frameon=False,
    fontsize=10,
    loc="upper right",
)

# Axis styling
ax.set_ylabel(
    "Protein ID",
    fontsize=20,
    fontweight="bold",
    color="#2a2a2a",
    labelpad=12,
)

ax.set_ylim(0, 7000)
ax.set_yticks(range(0, 6001, 1000))

ax.tick_params(
    axis="y",
    colors="#2a2a2a",
    labelsize=13,
    length=5,
    width=1.5,
    direction="out",
)

ax.tick_params(
    axis="x",
    colors="#2a2a2a",
    labelsize=13,
    length=5,
    width=1.5,
    direction="out",
)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_color("#2a2a2a")
ax.spines["bottom"].set_color("#2a2a2a")
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

ax.grid(False)

plt.tight_layout()

# ============================================================
# Save outputs
# ============================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

svg_out = Path(OUTPUT_DIR) / "protein_id_boxplot_with_quantifiable_diamond.svg"
png_out = Path(OUTPUT_DIR) / "protein_id_boxplot_with_quantifiable_diamond.png"

fig.savefig(svg_out, transparent=True, bbox_inches="tight")
fig.savefig(png_out, transparent=True, bbox_inches="tight")

plt.close(fig)

print(f"Saved SVG: {svg_out}")
print(f"Saved PNG: {png_out}")