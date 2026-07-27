"""ESI dGNN parity figure: four-panel comparison of dGNN predictions
against SCAN+U Ev.

  (a) Base model:  per-fold E_{k,v} (gray) and ensemble-mean (blue)
                   dGNN predictions vs SCAN+U Ev on the 90 CCTM sites.
  (b) FT model:    same, ensemble-mean in orange, on the 72-site hold-out.
  (c) {MAE}_K:     per-fold MAE distribution as boxplot; star marks
                   ensemble-mean overall MAE.
  (d) {R^2}_K:     per-fold Pearson r^2 distribution; star marks
                   ensemble-mean overall Pearson r^2.

Per-fold metrics for Base use the 90-site set; per-fold metrics for FT
use the 72-site hold-out.

Reads:  data/dgnn_preds_pretrained.csv, data/dgnn_preds_finetuned.csv,
        data/dgnn_cv_folds.csv, data/dgnn_cv_overall.csv
Writes: plots/figureS_dgnn_parity.png
Run from inside figures/: python figureS_dgnn_parity.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_dgnn_parity.png'

EES_DOUBLE_CM = fig_style.cm2in(fig_style.EES_DOUBLE_CM)

C_BASE = fig_style.COLOR_DGNN
C_FT   = fig_style.COLOR_DGNN_FT
C_FOLD = '#cfcfcf'    # gray for {E_{k,v}}

# figure-specific overrides
plt.rcParams.update({
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})


# === Load Base, FT, and per-fold CV summary ===
base    = pd.read_csv(DATA_DIR / 'dgnn_preds_pretrained.csv')
ft      = pd.read_csv(DATA_DIR / 'dgnn_preds_finetuned.csv')
cvs     = pd.read_csv(DATA_DIR / 'dgnn_cv_folds.csv')
cvs_all = pd.read_csv(DATA_DIR / 'dgnn_cv_overall.csv')

fold_cols = [f'DefCv{i}' for i in range(10)]


# === Panel (a)/(b) scatter data ===
def parity_arrays(df):
    """Return (x_dft, y_ens, x_fold_flat, y_fold_flat)."""
    x = df['DFT'].values
    y = df['avgDefCv'].values
    folds = df[fold_cols].values            # (N, 10)
    x_flat = np.repeat(x, folds.shape[1])
    y_flat = folds.ravel()
    return x, y, x_flat, y_flat


x_b, y_b, xf_b, yf_b = parity_arrays(base)
x_f, y_f, xf_f, yf_f = parity_arrays(ft)


# === Shared axis range across (a) and (b) ===
all_x = np.r_[x_b, x_f, xf_b, xf_f]
all_y = np.r_[y_b, y_f, yf_b, yf_f]
lo = min(all_x.min(), all_y.min())
hi = max(all_x.max(), all_y.max())
axmin = np.floor(lo * 2) / 2 - 0.25
axmax = np.ceil(hi * 2) / 2 + 0.25


# === Figure ===
W = EES_DOUBLE_CM
H = W * 0.31
fig = plt.figure(figsize=(W, H), dpi=600)
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.45, 0.45],
                      wspace=0.40, left=0.07, right=0.99,
                      top=0.86, bottom=0.20)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[0, 3])


def style_parity(ax, color_ens, fold_xy, ens_xy, panel_letter, panel_title):
    xfold, yfold = fold_xy
    xens,  yens  = ens_xy
    # Identity line
    ax.plot([axmin, axmax], [axmin, axmax],
            color='k', linestyle='--', linewidth=0.7, zorder=0)
    # Fold-level cloud
    ax.scatter(xfold, yfold, s=14, c=C_FOLD, edgecolor='none',
               alpha=0.55, zorder=1, rasterized=True,
               label=r'$\{E_{k,\mathrm{v}}\}$')
    # Ensemble means
    ax.scatter(xens, yens, s=22, c=color_ens, edgecolor='black',
               linewidth=0.35, alpha=0.95, zorder=2,
               label=r'$\hat{E}_\mathrm{v}$')
    ax.set_xlim(axmin, axmax); ax.set_ylim(axmin, axmax)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(np.ceil(axmin), np.floor(axmax) + 1e-6, 1.0))
    ax.set_yticks(np.arange(np.ceil(axmin), np.floor(axmax) + 1e-6, 1.0))
    ax.set_xlabel(r'SCAN+$U$ $E_\mathrm{v}$ (eV)')
    ax.set_title(rf'$\mathbf{{({panel_letter})}}$ {panel_title}', loc='left', pad=4)
    leg = ax.legend(loc='lower right', frameon=True, fontsize=6.5,
                    handletextpad=0.2, borderpad=0.3,
                    edgecolor='0.7', facecolor='white', framealpha=0.9)
    leg.get_frame().set_linewidth(0.5)


style_parity(ax_a, C_BASE, (xf_b, yf_b), (x_b, y_b), 'a', 'Base model')
style_parity(ax_b, C_FT,   (xf_f, yf_f), (x_f, y_f), 'b', 'FT model')
ax_a.set_ylabel(r'dGNN $E_\mathrm{v}$ (eV)')


# === Panels (c) and (d): per-fold boxplots with overall stars ===
mae_base = cvs['folds_base_MAE'].dropna().values
mae_ft   = cvs['folds_FT_MAE'].dropna().values
r2_base  = cvs['folds_base_R2'].dropna().values
r2_ft    = cvs['folds_FT_R2'].dropna().values

mae_all_base = cvs_all['all_base_MAE'].iloc[0]
mae_all_ft   = cvs_all['all_FT_MAE'].iloc[0]
r2_all_base  = cvs_all['all_base_R2'].iloc[0]
r2_all_ft    = cvs_all['all_FT_R2'].iloc[0]


def style_box(ax, base_data, ft_data, base_star, ft_star, ylabel, panel_letter):
    ax.boxplot([base_data, ft_data], positions=[1, 2],
               widths=0.55, patch_artist=False,
               medianprops=dict(color='#e6730d', linewidth=0.9),
               whiskerprops=dict(color='black', linewidth=0.6),
               capprops=dict(color='black', linewidth=0.6),
               boxprops=dict(color='black', linewidth=0.7),
               flierprops=dict(marker='o', markersize=4,
                               markerfacecolor='white',
                               markeredgecolor='black',
                               markeredgewidth=0.6))
    # Star: ensemble-mean overall metric on the full hold-out per model.
    ax.scatter([1], [base_star], marker='*', s=85, c=C_BASE,
               edgecolor='black', linewidth=0.4, zorder=3)
    ax.scatter([2], [ft_star],   marker='*', s=85, c=C_FT,
               edgecolor='black', linewidth=0.4, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Base', 'FT'])
    ax.set_ylabel(ylabel)
    ax.set_title(rf'$\mathbf{{({panel_letter})}}$', loc='left', pad=4)
    ax.set_xlim(0.4, 2.6)


style_box(ax_c, mae_base, mae_ft, mae_all_base, mae_all_ft,
          r'$\{\mathrm{MAE}\}_K$ (eV)', 'c')
style_box(ax_d, r2_base, r2_ft, r2_all_base, r2_all_ft,
          r'$\{R^2\}_K$', 'd')


fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Base star  (MAE, R2): {mae_all_base:.3f}, {r2_all_base:.3f}")
print(f"FT   star  (MAE, R2): {mae_all_ft:.3f}, {r2_all_ft:.3f}")
print(f"Wrote {OUT_PNG}")
