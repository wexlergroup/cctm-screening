"""Figure 2: bulk thermodynamic stability (E_hull) across the sampled composition space.

E_hull vs X_Mn for both compositional series (X_Mn = 2 X_Ce and X_Ce = 1/3),
anchored at the CCTM2112 intersection, with the 35 meV/atom synthesizability
threshold and finite-temperature (k_B T and k_B T + T dS_mix) reference lines.
Series A rises at about 83 meV per unit X_Mn and Series B at about 21 meV
per unit X_Mn, so most of the E_hull change reflects Ce variation, not Mn
variation, around the canonical point.

Reads:
    data/dft_ehull.csv
Writes:
    plots/figure2.png
Run from inside figures/: python3 figure2.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')

DATA_CSV = DATA_DIR / 'dft_ehull.csv'
OUT_PNG = PLOT_DIR / 'figure2.png'


# ---------- Load ----------
df = pd.read_csv(DATA_CSV)
keep = ['M', 'N', 'composition', 'x_Ce', 'x_Mn', 'ehull_in_meV', 'group']
df = df[keep].copy()

# Include intersection in both series so the connecting lines pass through it
df_A = df[df['group'].isin(['series_A', 'intersection'])].sort_values('x_Mn')
df_B = df[df['group'].isin(['series_B', 'intersection'])].sort_values('x_Mn')


# ---------- Plot ----------
W = fig_style.cm2in(fig_style.EES_SINGLE_CM)
fig, ax = plt.subplots(figsize=(W, W * 0.78), dpi=600)

c_A, c_B, c_X = '#1f6cb5', '#c0392b', '#e8b923'

# Series A line + circles
ax.plot(df_A['x_Mn'], df_A['ehull_in_meV'],
        color=c_A, lw=0.9, alpha=0.6, zorder=1)
sA = df_A[df_A['group'] == 'series_A']
ax.scatter(sA['x_Mn'], sA['ehull_in_meV'],
           c=c_A, marker='o', s=42, edgecolors='black', linewidths=0.6, zorder=3,
           label=r'$X_\mathrm{Mn} = 2\,X_\mathrm{Ce}$')

# Series B line + squares
ax.plot(df_B['x_Mn'], df_B['ehull_in_meV'],
        color=c_B, lw=0.9, alpha=0.6, zorder=1)
sB = df_B[df_B['group'] == 'series_B']
ax.scatter(sB['x_Mn'], sB['ehull_in_meV'],
           c=c_B, marker='s', s=42, edgecolors='black', linewidths=0.6, zorder=3,
           label=r'$X_\mathrm{Ce} = 1/3$')

# Intersection (CCTM2112)
inter = df[df['group'] == 'intersection']
ax.scatter(inter['x_Mn'], inter['ehull_in_meV'],
           c=c_X, marker='D', s=70, edgecolors='black', linewidths=0.6, zorder=4,
           label='CCTM2112')

ax.set_xlabel(r'$X_\mathrm{Mn}$')
ax.set_ylabel(r'$E_\mathrm{hull}$ (meV / atom)')
ax.set_xlim(0.46, 0.79)
ax.set_xticks([0.5, 0.6, 0.7])


# ---------- Reference: synthesizability threshold + finite-T offsets ----------
KB = 8.617333e-2          # k_B in meV/K
T_K = 298                 # K
KBT = KB * T_K            # 25.68 meV/atom
E_THRESH = 35.0           # empirical synthesizability threshold (meV/atom)


def TdS_mix(xCa, xCe, xTi, xMn, T=T_K):
    """T·ΔS_mix per crystallographic atom for ABO3 (5 atoms / f.u.).
    Binary mixing on the A-site (Ca/Ce) and on the B-site (Ti/Mn);
    the O sublattice does not mix."""
    def xlogx(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, x * np.log(x), 0.0)
    return -KB * T * (xlogx(xCa) + xlogx(xCe)
                      + xlogx(xTi) + xlogx(xMn)) / 5.0


# Series A: X_Mn = 2 X_Ce; both A- and B-site mixing entropies vary
xMn_A = np.linspace(0.50, 0.75, 200)
xCe_A = xMn_A / 2
TdS_A = TdS_mix(1 - xCe_A, xCe_A, 1 - xMn_A, xMn_A)

# Series B: X_Ce = 1/3 fixed; only B-site mixing entropy varies
xMn_B = np.linspace(0.5833, 0.75, 200)
xCe_B = np.full_like(xMn_B, 1 / 3)
TdS_B = TdS_mix(1 - xCe_B, xCe_B, 1 - xMn_B, xMn_B)

# Reference horizontal lines: synthesizability threshold and k_B T
ax.axhline(E_THRESH, color='0.30', lw=0.7, ls='--', zorder=1)
ax.axhline(KBT, color='0.55', lw=0.7, ls=':', zorder=1)

# Composition-dependent curves at k_B T + T·ΔS_mix (per series)
ax.plot(xMn_A, KBT + TdS_A,
        color=c_A, lw=0.8, ls='--', alpha=0.85, zorder=1)
ax.plot(xMn_B, KBT + TdS_B,
        color=c_B, lw=0.8, ls=(0, (4, 1.5, 1, 1.5)), alpha=0.85, zorder=1)

# Inline annotations: "35 meV/atom" and "k_B T + T·ΔS_mix" on the left
# (away from the X_Mn=0.75 data points); "k_B T" stays on the right.
ax.text(0.475, E_THRESH + 0.5, '35 meV/atom', va='bottom', ha='left',
        fontsize=6.5, color='0.30')
ax.text(0.785, KBT - 0.7, r'$k_\mathrm{B}T$',
        va='top', ha='right', fontsize=6.5, color='0.55')
# Anchored like the k_B T label: top edge 0.7 meV below the line it labels
# (the blue dashed curve's lowest point over the label's x-span is its left
# end at x_Mn = 0.50).
ax.text(0.475, (KBT + TdS_A[0]) - 0.7, r'$k_\mathrm{B}T + T\Delta S_\mathrm{mix}$',
        va='top', ha='left', fontsize=6.5, color='0.30')

ax.set_ylim(14, 43)

leg = ax.legend(loc='lower right', frameon=True, framealpha=0.95,
                edgecolor='0.6', borderpad=0.4, handletextpad=0.5,
                handlelength=1.6)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
print(f"Wrote {OUT_PNG}")
