"""ESI dGNN vs CFM screening-distribution figure: 10x10 histogram facet
comparing per-site fine-tuned dGNN Ev predictions to CFM-predicted Ev
across the (X_Ce, X_Mn) screening grid.

CFM evaluation uses SCAN+U features and the fractional-Mn assignment
from charge balance (n_Mn = 4 - X_Ce / X_Mn). Atomic property values
and canonical coefficients are taken from figureS_cfm_parity.py
(beta_b = +2.00, beta_r = -1.34, c = -4.26). The CFM is evaluated on
the full grid; in the X_Ce > X_Mn half-plane n_Mn extrapolates below
3 (charge balance broken), so the figure shows where the CFM's
predictions become unphysical outside its training domain.

Reads:  data/dgnn_screen_finetuned.csv
Writes: plots/figureS_dgnn_vs_cfm.png
Run from inside figures/: python figureS_dgnn_vs_cfm.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_dgnn_vs_cfm.png'

EES_DOUBLE_CM = fig_style.cm2in(fig_style.EES_DOUBLE_CM)

C_DGNN = fig_style.COLOR_DGNN    # filled blue
C_CFM  = fig_style.COLOR_CFM     # red outline

# figure-specific overrides
plt.rcParams.update({
    'axes.titlesize': 8,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})


# === CFM parameters (SCAN+U features, fractional Mn) ===
EB = {'Ca2': 1.8826, 'Ce3': 2.4817, 'Ti4': 3.1606,
      'Mn3': 1.9680, 'Mn4': 2.2541}
VR = {'Ca2_0': -3.2776, 'Ce3_0': -2.9510, 'Ti4_3': -1.9114,
      'Mn3_2': -1.0220, 'Mn4_3': -0.3965}
BETA_B, BETA_R, C0 = 2.00, -1.34, -4.26


def cfm_ev(X_Ce, X_Mn, xCe, xMn):
    """Per-site CFM Ev. X_Ce, X_Mn bulk; xCe, xMn local NN."""
    # Bulk n_Mn from charge balance; placeholder at X_Mn = 0 since the
    # Mn terms below have an x_Mn = 0 prefactor anyway.
    n = np.where(X_Mn > 0, 4.0 - X_Ce / np.maximum(X_Mn, 1e-12), 3.5)
    em = (n - 3) * EB['Mn4']   + (4 - n) * EB['Mn3']
    vm = (n - 3) * VR['Mn4_3'] + (4 - n) * VR['Mn3_2']
    Eb = (4*(1 - xCe)*EB['Ca2']  + 4*xCe*EB['Ce3']
          + 2*(1 - xMn)*EB['Ti4'] + 2*xMn*em) / 6
    Vr = (4*(1 - xCe)*VR['Ca2_0']  + 4*xCe*VR['Ce3_0']
          + 2*(1 - xMn)*VR['Ti4_3'] + 2*xMn*vm) / 6
    return C0 + BETA_B * Eb + BETA_R * Vr


# === Load per-site screen file (with bulk + local composition + dGNN) ===
df = pd.read_csv(DATA_DIR / 'dgnn_screen_finetuned.csv')

df['CFM_Ev'] = cfm_ev(df['X_Ce'].values, df['X_Mn'].values,
                      df['x_Ce_in_NN'].values, df['x_Mn_in_NN'].values)


# === Histogram setup: shared bins across all panels ===
ev_all = np.r_[df['avgDefCv'].values, df['CFM_Ev'].values]
ev_lo = np.floor(ev_all.min() * 2) / 2
ev_hi = np.ceil(ev_all.max() * 2) / 2
bins = np.linspace(ev_lo, ev_hi, 31)

xce_vals = np.sort(df['X_Ce'].unique())
xmn_vals = np.sort(df['X_Mn'].unique())


# === 10x10 facet (X_Ce descending top-to-bottom, X_Mn ascending L-to-R) ===
W = EES_DOUBLE_CM
H = W   # square panel block
fig, axes = plt.subplots(10, 10, figsize=(W, H), dpi=600,
                         sharex=True, sharey=True,
                         gridspec_kw=dict(wspace=0.10, hspace=0.10))

for i, xce in enumerate(xce_vals[::-1]):
    for j, xmn in enumerate(xmn_vals):
        ax = axes[i, j]
        sub = df[(df['X_Ce'] == xce) & (df['X_Mn'] == xmn)]
        if len(sub) > 0:
            ax.hist(sub['avgDefCv'].values, bins=bins,
                    color=C_DGNN, edgecolor='none', alpha=0.45)
            ax.hist(sub['CFM_Ev'].values, bins=bins,
                    histtype='step', color=C_CFM, linewidth=0.9)
        ax.set_yscale('log')
        ax.set_ylim(0.8, 600)
        ax.set_xlim(ev_lo, ev_hi)
        if i == 0:
            ax.set_title(rf'$X_\mathrm{{Mn}} = {xmn:.2f}$',
                         fontsize=7, pad=3)
        if j == len(xmn_vals) - 1:
            ax.text(1.06, 0.5, rf'$X_\mathrm{{Ce}} = {xce:.2f}$',
                    transform=ax.transAxes, rotation=270,
                    va='center', ha='left', fontsize=7)
        if j != 0:
            ax.tick_params(axis='y', labelleft=False)
        if i != 9:
            ax.tick_params(axis='x', labelbottom=False)


# === Shared axis labels and legend ===
# Measured placement: shared labels sit a fixed 4 pt pad from the outermost
# tick labels, and the legend's bottom edge sits above the top-row column
# titles so the frame cannot overlap them.
fig.subplots_adjust(top=0.93)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
pad_px = 4 / 72 * fig.dpi

ticks_y0 = min(t.get_window_extent(renderer).y0
               for ax in axes[9, :] for t in ax.get_xticklabels()
               if t.get_text())
fig.supxlabel(r'$\hat{E}_\mathrm{v}$ (eV)',
              y=(ticks_y0 - pad_px) / fig.bbox.height, va='top', fontsize=9)

ticks_x0 = min(t.get_window_extent(renderer).x0
               for ax in axes[:, 0] for t in ax.get_yticklabels()
               if t.get_text())
fig.supylabel('counts',
              x=(ticks_x0 - pad_px) / fig.bbox.width, ha='right', fontsize=9)

legend_handles = [
    Patch(facecolor=C_DGNN, edgecolor='none', alpha=0.45, label='dGNN (FT)'),
    Line2D([0], [0], color=C_CFM, linewidth=1.4, label='CFM'),
]
titles_y1 = max(ax.title.get_window_extent(renderer).y1 for ax in axes[0, :])
fig.legend(handles=legend_handles, loc='lower center',
           bbox_to_anchor=(0.5, (titles_y1 + pad_px) / fig.bbox.height),
           frameon=False,
           ncol=2, handlelength=1.8, columnspacing=1.4,
           fontsize=8)

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Bins span {ev_lo:.2f} to {ev_hi:.2f} eV")
print(f"dGNN range: {df['avgDefCv'].min():.2f} to {df['avgDefCv'].max():.2f}")
print(f"CFM  range: {df['CFM_Ev'].min():.2f}  to {df['CFM_Ev'].max():.2f}")
print(f"Wrote {OUT_PNG}")
