"""Fig. 5: predicted redox capacity over (X_Ce, X_Mn) composition space.

Two panels, CFM (trained on DFT) and dGNN, on a shared color scale so the
predicted Delta-delta landscapes can be compared directly. Markers indicate
the six DFT-sampled bulk compositions, color-coded by series (matching
Figs. 1c, 2, 3, 4): blue circles = Series A (X_Mn = 2 X_Ce); red squares =
Series B (X_Ce = 1/3); gold diamond = CCTM2112 (intersection). The dGNN
gives slightly negative Delta-delta at the high-X_Mn / low-X_Ce edge
(extrapolation artifact); values are clipped to 0 in the colormap.

Reads: data/cfm_delta_grid.csv, data/dgnn_delta_grid.csv
Writes: plots/figure5.png
Run from inside figures/: python3 figure5.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')

CFM_CSV = DATA_DIR / 'cfm_delta_grid.csv'
DGNN_CSV = DATA_DIR / 'dgnn_delta_grid.csv'
OUT_PNG = PLOT_DIR / 'figure5.png'

# figure-specific overrides
plt.rcParams.update({
    'legend.fontsize': 6.5,
})


# ---------- Load ----------
df_cfm = pd.read_csv(CFM_CSV)
df_dgnn = pd.read_csv(DGNN_CSV)


def to_grid(df):
    return df.pivot(index='X_Mn', columns='X_Ce', values='delta_delta')


g_cfm = to_grid(df_cfm)
g_dgnn = to_grid(df_dgnn)

xCe_axis = np.array(sorted(df_cfm['X_Ce'].unique()))
xMn_axis = np.array(sorted(df_cfm['X_Mn'].unique()))
X, Y = np.meshgrid(xCe_axis, xMn_axis)


# ---------- Bulk compositions to overlay (matching Figs 1c, 2, 3, 4) ----------
c_A, c_B, c_X = '#1f6cb5', '#c0392b', '#e8b923'
bulks = [
    # (X_Ce, X_Mn, color, marker)
    (0.250, 0.500, c_A, 'o'),    # M=0,  N=18  (Series A)
    (0.292, 0.583, c_A, 'o'),    # M=0,  N=21  (Series A)
    (0.375, 0.750, c_A, 'o'),    # M=0,  N=27  (Series A)
    (0.333, 0.583, c_B, 's'),    # M=6,  N=24  (Series B)
    (0.333, 0.750, c_B, 's'),    # M=-6, N=24  (Series B)
    (0.333, 0.667, c_X, 'D'),    # M=0,  N=24  (CCTM2112, intersection)
]


# ---------- Plot ----------
W = fig_style.cm2in(fig_style.EES_DOUBLE_CM)
fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.50), dpi=600,
                         sharey=True,
                         gridspec_kw={'wspace': 0.10, 'right': 0.86})

# Common color scale: clip negatives to 0
vmax = max(g_cfm.values.max(), g_dgnn.values.max())
norm = Normalize(vmin=0.0, vmax=vmax)
cmap = plt.cm.viridis
isolines = np.arange(0.05, vmax, 0.05)

panels = [
    (axes[0], np.clip(g_cfm.values, 0, None), 'CFM', '(a)'),
    (axes[1], np.clip(g_dgnn.values, 0, None), 'dGNN', '(b)'),
]

# One contour label per panel collides with the composition markers when
# placed automatically (the CFM 0.05 line passes through CCTM2112; the dGNN
# 0.15 line kinks at the Series B marker pair). Those levels get manual
# label positions on marker-free segments of the same contour.
MANUAL_CLABELS = {
    'CFM': (0.05, (0.09, 0.45)),
    'dGNN': (0.15, (0.50, 0.865)),
}

for ax, Z, title, letter in panels:
    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap, norm=norm)
    cs = ax.contour(X, Y, Z, levels=isolines,
                    colors='white', linewidths=0.5, alpha=0.7)
    manual_lvl, manual_pos = MANUAL_CLABELS[title]
    is_manual = np.isclose(cs.levels, manual_lvl)
    ax.clabel(cs, levels=cs.levels[~is_manual], inline=True,
              fontsize=6, fmt='%.2f')
    # No levels= here: restricting levels together with manual positions
    # trips an index bug in matplotlib's add_label_near; the position sits
    # exactly on the intended contour, so nearest-contour matching finds it.
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f', manual=[manual_pos])

    # Overlay sampled bulks
    for xc, xm, color, marker in bulks:
        size = 55 if marker == 'D' else 40
        ax.scatter(xc, xm, c=color, marker=marker, s=size,
                   edgecolors='white', linewidths=0.9, zorder=5)

    ax.set_title(title, fontsize=8.5, pad=4)
    # Bold panel letter in the upper-left corner (placement matches figure1.py)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

axes[0].set_ylabel(r'$X_\mathrm{Mn}$')
for ax in axes:
    ax.set_xlabel(r'$X_\mathrm{Ce}$')

# Shared colorbar
cbar_ax = fig.add_axes([0.88, 0.18, 0.016, 0.66])
cbar = fig.colorbar(cf, cax=cbar_ax)
cbar.set_label(r'$\Delta\delta$', fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
print(f"Wrote {OUT_PNG}")
