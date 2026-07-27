"""ESI dGNN screening summary figure: mean and standard deviation of
per-site dGNN-predicted Ev across the (X_Ce, X_Mn) grid, plotted as
line traces in X_Mn with one curve per X_Ce.

Reads:  data/dgnn_screen_finetuned.csv
Writes: plots/figureS_dgnn_summary.png
Run from inside figures/: python figureS_dgnn_summary.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_dgnn_summary.png'

EES_DOUBLE_CM = fig_style.cm2in(fig_style.EES_DOUBLE_CM)

# figure-specific overrides
plt.rcParams.update({
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})


# === Load FT screening predictions (pre-aggregated: X_Ce, X_Mn, avgDefCv) ===
ft = pd.read_csv(DATA_DIR / 'dgnn_screen_finetuned.csv')

agg = (ft.groupby(['X_Ce', 'X_Mn'])['avgDefCv']
         .agg(mean='mean', std='std')
         .reset_index())


# === Plot ===
xce_vals = np.sort(agg['X_Ce'].unique())
# Color the composition (X_Ce) axis by sampling viridis, matching the
# local-x_Ce convention in figure4.py (avoiding near-white extremes).
cmap = plt.cm.viridis
colors = {xc: cmap(0.10 + 0.80 * i / (len(xce_vals) - 1))
          for i, xc in enumerate(xce_vals)}

W = EES_DOUBLE_CM
H = W * 0.42
fig, axes = plt.subplots(1, 2, figsize=(W, H), dpi=600,
                         gridspec_kw=dict(wspace=0.28))

for ax_i, (col, ylab) in enumerate([
    ('mean', r'$\langle \hat{E}_\mathrm{v} \rangle_\mathrm{SQS}$ (eV)'),
    ('std',  r'$\sigma_\mathrm{SQS}(\hat{E}_\mathrm{v})$ (eV)'),
]):
    ax = axes[ax_i]
    for xce in xce_vals:
        sub = agg[agg['X_Ce'] == xce].sort_values('X_Mn')
        ax.plot(sub['X_Mn'], sub[col], color=colors[xce],
                marker='s', markersize=4.5, markeredgecolor='none',
                linewidth=1.1, label=f'{xce:.3f}')
    ax.set_xlabel(r'$X_\mathrm{Mn}$')
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 1.01, 0.25))

# Single shared legend across both panels, its bottom edge anchored just
# above the axes so the frame cannot overlap either panel.
handles, labels = axes[0].get_legend_handles_labels()
axes_top = max(ax.get_position().y1 for ax in axes)
fig.legend(handles, labels, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, axes_top + 0.02), frameon=False,
           title=r'$X_\mathrm{Ce}$', title_fontsize=7.5,
           handletextpad=0.3, columnspacing=1.2)

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Wrote {OUT_PNG}")
