# -*- coding: utf-8 -*-
"""
RT stability plot: scatter + odd/even separated boxplot
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['svg.fonttype'] = 'none'

# Global font settings
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18

# Read data
df = pd.read_csv("RT_4pep.csv", header=None)

# Data
x = pd.to_numeric(df.iloc[:, 0], errors="coerce")

series = [
    pd.to_numeric(df.iloc[:, i], errors="coerce")
    for i in range(1, 5)
]

# Remove invalid rows
valid_x = ~x.isna()
x = x[valid_x]
series = [s[valid_x] for s in series]

# Odd / Even injections
odd_mask = x % 2 == 1
even_mask = x % 2 == 0

# Colors
colors = {
    "odd": "#1f77b4",
    "even": "#ff7f0e"
}

# Sort peptides by median RT
medians = [s.median() for s in series]
sorted_idx = sorted(
    range(len(series)),
    key=lambda i: medians[i]
)

odd_sorted = [series[i][odd_mask] for i in sorted_idx]
even_sorted = [series[i][even_mask] for i in sorted_idx]

# ==========================================
# Figure size: 20 × 7 inch
# ==========================================
fig = plt.figure(figsize=(20, 7))

# ==========================================
# Left panel: scatter plot
# ==========================================
ax1 = fig.add_axes([0.06, 0.12, 0.78, 0.78])

for y in series:

    ax1.scatter(
        x[odd_mask],
        y[odd_mask],
        s=8,
        color=colors["odd"],
        alpha=0.7
    )

    ax1.scatter(
        x[even_mask],
        y[even_mask],
        s=8,
        color=colors["even"],
        alpha=0.7
    )

ax1.set_xlim(0, 1200)
ax1.set_ylim(0, 5)

ax1.set_yticks([0, 1, 2, 3, 4, 5])

ax1.set_xlabel("Number of Injections")
ax1.set_ylabel("Retention Time (min)")

# Axis styling
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

ax1.tick_params(
    axis='both',
    which='major',
    width=1.5,
    length=6,
    labelsize=18
)

# Legend
legend_elements = [

    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='Odd injection',
        markerfacecolor=colors["odd"],
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='Even injection',
        markerfacecolor=colors["even"],
        markersize=8
    )
]

ax1.legend(
    handles=legend_elements,
    loc='upper right',
    frameon=False
)

# ==========================================
# Right panel: boxplot
# ==========================================
ax2 = fig.add_axes(
    [0.86, 0.12, 0.10, 0.78],
    sharey=ax1
)

positions_odd = [1, 4, 7, 10]
positions_even = [2, 5, 8, 11]

bp_odd = ax2.boxplot(
    odd_sorted,
    positions=positions_odd,
    widths=0.6,
    vert=True,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    medianprops=dict(
        linewidth=1.5,
        color="black"
    )
)

bp_even = ax2.boxplot(
    even_sorted,
    positions=positions_even,
    widths=0.6,
    vert=True,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    medianprops=dict(
        linewidth=1.5,
        color="black"
    )
)

for box in bp_odd["boxes"]:
    box.set_facecolor(colors["odd"])
    box.set_alpha(0.7)

for box in bp_even["boxes"]:
    box.set_facecolor(colors["even"])
    box.set_alpha(0.7)

# Hide right-side axis
ax2.set_xticks([])

ax2.tick_params(
    axis='y',
    left=False,
    labelleft=False
)

for spine in ax2.spines.values():
    spine.set_visible(False)

# Save
plt.savefig(
    "final_plot_odd_even_boxplot.svg",
    format="svg",
    bbox_inches="tight"
)

plt.savefig(
    "final_plot_odd_even_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()