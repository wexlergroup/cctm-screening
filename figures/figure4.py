"""Fig. 4: oxygen-vacancy formation energy vs local 1NN composition.

For each (x_Ce, x_Mn) cell of the 1NN environment of an oxygen vacancy, the
six bulk values are plotted as small markers (deterministic horizontal
jitter). Cell means (Count-weighted across bulks) are shown as larger
markers and connected by lines within each x_Ce series. The lines show the
dominant B-site slope (~1.3 eV across x_Mn), the vertical separation between
colors at each x_Mn shows the secondary A-site modulation (~0.55 eV across
x_Ce), and the tight clustering of the small markers within each cell
supports the universality of the local-environment dependence across bulk
compositions. The shaded band marks the target E_v window (3.4-3.9 eV) for
solar thermochemical H2 production.

Reads: data/dft_ev_local_env.csv
Writes: plots/figure4.png
Run from inside figures/: python3 figure4.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')

EV_CSV = DATA_DIR / 'dft_ev_local_env.csv'
OUT_PNG = PLOT_DIR / 'figure4.png'
TARGET_EV = (3.4, 3.9)

# figure-specific overrides
plt.rcParams.update({
    'legend.fontsize': 6.5,
})


# ---------- Load ----------
df = pd.read_csv(EV_CSV)
keep = ['Composition', 'M', 'N', 'XCe', 'XMn', 'xCe', 'xMn', 'Count', 'Ev']
df = df[keep].copy()

xCe_values = sorted(df['xCe'].unique())   # [0, 1/4, 1/2, 3/4, 1]
xMn_values = sorted(df['xMn'].unique())   # [0, 1/2, 1]

# Discrete colors from viridis, avoiding near-white extremes
cmap = plt.cm.viridis
colors_xCe = {x: cmap(0.10 + 0.80 * i / (len(xCe_values) - 1))
              for i, x in enumerate(xCe_values)}


def x_position(xMn, xCe, span=0.16):
    """Cluster the 5 x_Ce groups within each x_Mn position."""
    return xMn + (xCe - 0.5) * span


# ---------- Plot ----------
W = fig_style.cm2in(fig_style.EES_SINGLE_CM)
fig, ax = plt.subplots(figsize=(W, W * 0.85), dpi=600)

# Target band (drawn first so violins/markers sit on top)
ax.axhspan(*TARGET_EV, color='0.92', zorder=0)

# Individual bulk values: small markers, deterministic horizontal jitter
for xMn in xMn_values:
    for xCe in xCe_values:
        sub = df[(df['xMn'] == xMn) & (df['xCe'] == xCe)]
        n = len(sub)
        x_c = x_position(xMn, xCe)
        if n > 1:
            x_jit = np.linspace(-0.012, 0.012, n)
        else:
            x_jit = [0.0]
        ax.scatter(x_c + np.array(x_jit), sub['Ev'].values,
                   color=colors_xCe[xCe], s=9, alpha=0.55,
                   edgecolors='none', zorder=2)

# Cell means (Count-weighted), connected within each x_Ce series
xCe_labels = {0.0: '0', 0.25: '1/4', 0.5: '1/2', 0.75: '3/4', 1.0: '1'}
for xCe in xCe_values:
    pts = []
    for xMn in xMn_values:
        sub = df[(df['xMn'] == xMn) & (df['xCe'] == xCe)]
        if len(sub) and sub['Count'].sum() > 0:
            ev_w = np.average(sub['Ev'], weights=sub['Count'])
            pts.append((x_position(xMn, xCe), ev_w))
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colors_xCe[xCe], lw=1.0, alpha=0.9,
                marker='o', markersize=5.5,
                markerfacecolor=colors_xCe[xCe], markeredgecolor='black',
                markeredgewidth=0.5, zorder=3,
                label=f'${xCe_labels[xCe]}$')

# Target-band label
ax.text(0.01, 0.5 * (TARGET_EV[0] + TARGET_EV[1]),
        'target', ha='left', va='center',
        fontsize=6.5, color='0.40',
        transform=ax.get_yaxis_transform())

# Axes
ax.set_xlabel(r'Local $x_\mathrm{Mn}$')
ax.set_ylabel(r'$E_\mathrm{v}$ (eV)')
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(['$0$', '$1/2$', '$1$'])
ax.set_xlim(-0.13, 1.13)
ax.set_ylim(2.5, 4.95)

leg = ax.legend(title=r'Local $x_\mathrm{Ce}$', loc='upper right',
                frameon=True, framealpha=0.95, edgecolor='0.6',
                borderpad=0.4, handletextpad=0.5,
                fontsize=6.5, title_fontsize=7)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
print(f"Wrote {OUT_PNG}")
