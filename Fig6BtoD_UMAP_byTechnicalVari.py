import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# --- 1. PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent

UMAP_FILE = BASE_DIR / "Input/Fig6+7_conc_SC_umap_cordi_post_iqr.csv"
FEATURE_FILE = BASE_DIR / "Input/Fig6BtoD_conc_sampleIdentity_forUMAP.csv"

OUTPUT_SVG = BASE_DIR / "output/Fig6+7/UMAP/NC_umap_batch.svg"
OUTPUT_PNG = BASE_DIR / "output/Fig6+7/UMAP/NC_umap_batch.png"

# --- AUTOMATICALLY CREATE THE OUTPUT FOLDERS ---
OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)

# --- 2. PLOT CONFIGURATION ---
COLOR_MODE = "category"  # "gradient" or "category"

# For gradient, use something like: COLOR_BY = "CISY"
# For category, use something like: COLOR_BY = "gate"
COLOR_BY = "plate" 

CUSTOM_COLORS = [
    "#440154",  # low values
    "#21918c",  # middle values
    "#fde725"   # high values
#Color for Gradient
]

CATEGORY_COLORS = {
    "GateP2": "#b41f87",
    "timeP2": "#1f77b4",
    "timeP3": "#ff7f0e",
    "timeP4": "#42b41f",
    
}

# --- AXIS RANGE CONFIG ---
AUTO_RANGE = False

X_LIMITS = (0, 13)
Y_LIMITS = (-3.5, 16)

# --- 3. LOAD AND PREPARE DATA ---
df_umap = pd.read_csv(UMAP_FILE)
df_feat = pd.read_csv(FEATURE_FILE)

df_umap.columns = df_umap.columns.str.strip()
df_feat.columns = df_feat.columns.str.strip()

df_umap["join_id"] = df_umap["sample"].astype(str).str.lower().str.strip()
df_feat["join_id"] = df_feat["sample"].astype(str).str.lower().str.strip()

df = pd.merge(df_umap, df_feat, on="join_id")

# --- 4. VISUALIZATION SETTINGS ---
mpl.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# --- 5. GENERATE PLOT ---
fig, ax = plt.subplots(figsize=(6, 5))

if COLOR_MODE == "gradient":
    custom_cmap = LinearSegmentedColormap.from_list("ms_gradient", CUSTOM_COLORS)

    scatter = ax.scatter(
        df["UMAP1"],
        df["UMAP2"],
        c=df[COLOR_BY],
        cmap=custom_cmap,
        s=15,
        edgecolors="none",
        alpha=0.75,
        # vmax=200000,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{COLOR_BY} Intensity", rotation=270, labelpad=15)

elif COLOR_MODE == "category":
    df[COLOR_BY] = df[COLOR_BY].astype(str).str.strip()

    for category, color in CATEGORY_COLORS.items():
        sub = df[df[COLOR_BY] == category]

        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            label=category,
            color=color,
            s=15,
            edgecolors="none",
            alpha=0.80,
        )

    ax.legend(title=COLOR_BY, frameon=False)

else:
    raise ValueError("COLOR_MODE must be either 'gradient' or 'category'")

# --- Apply axis limits ---
if not AUTO_RANGE:
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)

ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title(f"Cell Distribution: {COLOR_BY}", fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- 6. SAVE AND EXPORT ---
plt.tight_layout()
plt.savefig(OUTPUT_SVG, format="svg", dpi=300)
plt.savefig(OUTPUT_PNG, format="png", dpi=300)

print(f"Files saved:\n1. {OUTPUT_SVG}\n2. {OUTPUT_PNG}")
# plt.show()