"""CFM parity plots (random 50/50 and leave-one-composition-out) against the
SCAN+U site vacancy formation energies, with fractional-Mn descriptor
assignment and Huber regression.

Reads:  data/dft_ev_sites.csv (90 SCAN+U site energies)
        cfm-settings/features.csv (atomic descriptors, via cfm_model.load_species)
Writes: plots/figureS_cfm_parity.png (600 dpi, EES double column)

Run from inside figures/: python figureS_cfm_parity.py

The bootstrap, LOGO, and random-split statistics printed by this script are
seeded (default_rng(2024)) and quoted in the ESI; do not change any
computation or the RNG call order.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
from pathlib import Path

import fig_style
from cfm_model import load_species

# figure-specific overrides
plt.rcParams.update({'axes.titlesize': 8})

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_cfm_parity.png'

df = pd.read_csv(DATA_DIR / 'dft_ev_sites.csv')

# Atomic descriptor values (Eb = cation bond-dissociation energy, Vr = crystal
# reduction potential) are sourced from cfm-settings/features.csv via cfm_model,
# the single source shared with the Delta-delta screening. Mn3/Mn4 (Eb) and the
# Mn4->Mn3 / Mn3->Mn2 couples (Vr) are the endpoints of the fractional-Mn interpolation.
_species = load_species()
EB_DFT, EB_EXP = _species['EB_DFT'], _species['EB_EXP']
VR_DFT, VR_EXP = _species['VR_DFT'], _species['VR_EXP']

df['XCe_exact'] = df['NCe'] / df['NA']
df['XMn_exact'] = df['NMn'] / df['NA']
df['n_Mn'] = 4 - df['XCe_exact'] / df['XMn_exact']

def eb_mn_frac(n, src):
    e = EB_DFT if src == 'dft' else EB_EXP
    return (n - 3) * e['Mn4'] + (4 - n) * e['Mn3']
def vr_mn_frac(n, src):
    v = VR_DFT if src == 'dft' else VR_EXP
    return (n - 3) * v['Mn4_3'] + (4 - n) * v['Mn3_2']

def per_site(row, src):
    xCe, xMn, n = row['xCe'], row['xMn'], row['n_Mn']
    eb = EB_DFT if src == 'dft' else EB_EXP
    vr = VR_DFT if src == 'dft' else VR_EXP
    em = eb_mn_frac(n, src); vm = vr_mn_frac(n, src)
    Eb = (4*(1-xCe)*eb['Ca2'] + 4*xCe*eb['Ce3']
          + 2*(1-xMn)*eb['Ti4'] + 2*xMn*em) / 6
    Vr = (4*(1-xCe)*vr['Ca2_0'] + 4*xCe*vr['Ce3_0']
          + 2*(1-xMn)*vr['Ti4_3'] + 2*xMn*vm) / 6
    return Eb, Vr

dft_feats = df.apply(lambda r: per_site(r, 'dft'), axis=1, result_type='expand')
exp_feats = df.apply(lambda r: per_site(r, 'exp'), axis=1, result_type='expand')
df['Eb_dft'] = dft_feats[0]; df['Vr_dft'] = dft_feats[1]
df['Eb_exp'] = exp_feats[0]; df['Vr_exp'] = exp_feats[1]

X_dft = df[['Eb_dft', 'Vr_dft']].values
X_exp = df[['Eb_exp', 'Vr_exp']].values
y = df['Ev'].values
groups = df['Composition'].values

def fit_and_boot(X, y, n_boot=2000):
    m = HuberRegressor().fit(X, y)
    full = np.r_[m.intercept_, m.coef_]
    rng = np.random.default_rng(2024)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            mb = HuberRegressor().fit(X[idx], y[idx])
            boot.append(np.r_[mb.intercept_, mb.coef_])
        except Exception:
            pass
    return full, np.array(boot)

def random_cv(X, y, n=1000):
    rng = np.random.default_rng(2024)
    seeds = rng.integers(0, 10000, size=n)
    out = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=int(s))
        out.append(mean_absolute_error(yte, HuberRegressor().fit(Xtr, ytr).predict(Xte)))
    return np.array(out)

def logo(X, y, groups):
    yhat = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        yhat[te] = HuberRegressor().fit(X[tr], y[tr]).predict(X[te])
    return yhat

full_dft, boot_dft = fit_and_boot(X_dft, y)
full_exp, boot_exp = fit_and_boot(X_exp, y)
mae_rand_dft = random_cv(X_dft, y)
mae_rand_exp = random_cv(X_exp, y)
yhat_logo = logo(X_dft, y, groups)
mae_logo = mean_absolute_error(y, yhat_logo)
yhat_dft = HuberRegressor().fit(X_dft, y).predict(X_dft)
yhat_exp = HuberRegressor().fit(X_exp, y).predict(X_exp)

print("=== Coefficients (full-data Huber, fractional Mn) ===")
for label, full, boot in [('DFT', full_dft, boot_dft), ('EXP', full_exp, boot_exp)]:
    print(f"\n{label}:")
    for nm, v, b in zip(['c','beta_b','beta_r'], full, boot.T):
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {nm:<10} {v:>7.2f}  [{lo:>7.2f}, {hi:>7.2f}]")
print(f"\nMAE random 50/50: DFT={mae_rand_dft.mean():.3f}±{mae_rand_dft.std():.3f}, "
      f"EXP={mae_rand_exp.mean():.3f}±{mae_rand_exp.std():.3f}")
print(f"MAE LOGO (DFT): {mae_logo:.3f}")

comp_info = (df.groupby('Composition')
             .agg(NCe=('NCe','first'), NMn=('NMn','first'), NA=('NA','first'),
                  XCe=('XCe_exact','first'), XMn=('XMn_exact','first'))
             .reset_index().sort_values(['XCe','XMn']).reset_index(drop=True))
comp_labels = {}
for _, r in comp_info.iterrows():
    XCe, XMn = r['XCe'], r['XMn']
    XCa = 1 - XCe; XTi = 1 - XMn
    comp_labels[r['Composition']] = (
        f"Ca$_{{{XCa:.2f}}}$Ce$_{{{XCe:.2f}}}$"
        f"Ti$_{{{XTi:.2f}}}$Mn$_{{{XMn:.2f}}}$O$_3$"
    )
ordered_comps = comp_info['Composition'].tolist()
cmap = plt.get_cmap('viridis')
comp_colors = {c: cmap(i / (len(ordered_comps) - 1)) for i, c in enumerate(ordered_comps)}

axmin, axmax = 2.4, 5.0
def style_parity(ax, panel, label_text, mae_str):
    ax.plot([axmin, axmax], [axmin, axmax], 'k--', lw=0.7, alpha=0.6, zorder=0)
    ax.set_xlim(axmin, axmax); ax.set_ylim(axmin, axmax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'CFM $E_\mathrm{v}$ (eV)')
    ax.set_ylabel(r'SCAN+$U$ $E_\mathrm{v}$ (eV)')
    ax.set_title(f"{label_text}\n{mae_str}",
                 fontsize=8.5, pad=4)
    ax.text(0.04, 0.96, f'({panel})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

fig_b, axes_b = plt.subplots(1, 3, figsize=(fig_style.cm2in(fig_style.EES_DOUBLE_CM),
                                            fig_style.cm2in(7.5)))
axes_b[0].scatter(yhat_dft, y, s=18, c=fig_style.COLOR_CFM, edgecolor='white', linewidth=0.4,
                  alpha=0.85, zorder=2)
style_parity(axes_b[0], 'a', 'SCAN+$U$ features\nrandom 50/50',
             f"MAE = {mae_rand_dft.mean():.3f} $\\pm$ {mae_rand_dft.std():.3f} eV")
axes_b[1].scatter(yhat_exp, y, s=18, c=fig_style.COLOR_CFM, edgecolor='white', linewidth=0.4,
                  alpha=0.85, zorder=2)
style_parity(axes_b[1], 'b', 'Experimental features\nrandom 50/50',
             f"MAE = {mae_rand_exp.mean():.3f} $\\pm$ {mae_rand_exp.std():.3f} eV")
ax_c = axes_b[2]
for comp in ordered_comps:
    mask = groups == comp
    ax_c.scatter(yhat_logo[mask], y[mask], s=18, c=[comp_colors[comp]],
                 edgecolor='white', linewidth=0.4, alpha=0.9, zorder=2)
style_parity(ax_c, 'c', 'SCAN+$U$ features\nleave-one-comp.-out',
             f"MAE = {mae_logo:.3f} eV")
handles = [Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=comp_colors[c], markeredgecolor='white',
                  markeredgewidth=0.4, markersize=6, label=comp_labels[c])
           for c in ordered_comps]
fig_b.tight_layout(w_pad=1.5)
fig_b.subplots_adjust(bottom=0.16)

# Place the legend so the white space between the bottom of the x-axis
# labels and the legend top equals the space between the top of the x-axis
# labels and the bottom of the x-tick labels (measured from the canvas).
fig_b.canvas.draw()
renderer = fig_b.canvas.get_renderer()
xlab_bbs = [ax.xaxis.label.get_window_extent(renderer) for ax in axes_b]
xlab_y0 = min(bb.y0 for bb in xlab_bbs)
xlab_y1 = max(bb.y1 for bb in xlab_bbs)
ticks_y0 = min(t.get_window_extent(renderer).y0
               for ax in axes_b for t in ax.get_xticklabels())
gap_px = ticks_y0 - xlab_y1
leg_top_fig = (xlab_y0 - gap_px) / fig_b.bbox.height
fig_b.legend(handles=handles, loc='upper center', ncol=3,
             frameon=False, fontsize=7, handletextpad=0.3,
             columnspacing=1.5, bbox_to_anchor=(0.5, leg_top_fig))
fig_b.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig_b)
print(f"Wrote {OUT_PNG}")
