import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# --- 1. PATH CONFIGURATION (Modern Pathlib) ---
BASE_DIR = Path(__file__).resolve().parent

# Define input files relative to the script location
# Change these filenames to match your actual files
UMAP_FILE = BASE_DIR / "Input/Fig6+7_conc_SC_umap_cordi_post_iqr.csv"      
FEATURE_FILE = BASE_DIR /   "Input/Fig7BtoH_AllProteinIntensityForUMAP.csv"  #"LL2112_NC_0424/conc_sampleIdentity_forUMAP.csv"
                         

# Define output file
OUTPUT_SVG = BASE_DIR / "output/Fig6+7/UMAP/NC_umap_ANXA7.svg"
OUTPUT_PNG = BASE_DIR / "output/Fig6+7/UMAP/NC_umap_ANXA7.png"

# --- 2. PLOT CONFIGURATION ---
COLOR_BY = 'ANXA7'  # Header for the color gradient
# Define your custom 3-color sequence (or 2-color)
# Order: [LOW values, MIDDLE values, HIGH values]
CUSTOM_COLORS = [
    "#440154",  # Dark,
    "#21918c",  # Bright Orange 
    "#fde725"   # Pale Yellow 
]

# --- AXIS RANGE CONFIG ---
AUTO_RANGE = False  # True = matplotlib decides, False = use manual limits

# Only used if AUTO_RANGE = False
X_LIMITS = (0,13)
Y_LIMITS = (-3.5, 16)

# --- 3. LOAD AND PREPARE DATA ---
df_umap = pd.read_csv(UMAP_FILE)
df_feat = pd.read_csv(FEATURE_FILE)

# Standardize columns and merge
df_umap.columns = df_umap.columns.str.strip()
df_feat.columns = df_feat.columns.str.strip()

# Create a clean join ID (lowercase 'sample' vs 'Sample')
df_umap['join_id'] = df_umap['sample'].astype(str).str.lower().str.strip()
df_feat['join_id'] = df_feat['sample'].astype(str).str.lower().str.strip()

df = pd.merge(df_umap, df_feat, on='join_id')

# --- 4. VISUALIZATION SETTINGS ---
mpl.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

custom_cmap = LinearSegmentedColormap.from_list("ms_gradient", CUSTOM_COLORS)

# --- 5. GENERATE PLOT ---
fig, ax = plt.subplots(figsize=(6, 5))

scatter = ax.scatter(
    df['UMAP1'], 
    df['UMAP2'], 
    c=df[COLOR_BY], 
    cmap=custom_cmap, 
    s=15, 
    edgecolors='none', 
    alpha=0.75,
    # vmax=200000,
)

# --- Apply axis limits ---
if not AUTO_RANGE:
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_title(f'Cell Distribution: {COLOR_BY}', fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

cbar = plt.colorbar(scatter)
cbar.set_label(f'{COLOR_BY} Intensity', rotation=270, labelpad=15)

# --- 6. SAVE AND EXPORT ---
plt.tight_layout()
plt.savefig(OUTPUT_SVG, format='svg', dpi=300)
plt.savefig(OUTPUT_PNG, format='png', dpi=300)
print(f"Files saved:\n1. {OUTPUT_SVG}\n2. {OUTPUT_PNG}")
# plt.show()