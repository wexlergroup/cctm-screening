"""Thomas-Fermi screened vacancy-vacancy interaction V_vv(r), analytic.

Single panel: V_vv(r) for a hypothetical (2e)^2 pair, Thomas-Fermi screened
at lambda_TF derived from the mean Fermi-level DOS across the six bulk
compositions (see the ESI bulk-DOS figure, figureS_dos.py). Plotted at the
family-mean g(E_F) and at a conservative-low g(E_F) (half the mean).

The figure is organised around the statistical-mechanical model:
- Vertical band marks the experimentally accessed reduction-side delta
  range, with r mapped via the Wigner-Seitz random-distribution radius.
- Horizontal line marks k_B T at the TGA reduction peak temperature.
- A top secondary x-axis converts r to delta_red.
- The supercell periodic-image distance r_min ~ 15.4 A is marked as a
  light side-reference.

Reads:  nothing (analytic; all inputs are constants below)
Writes: plots/figureS_vv.png (600 dpi, EES single column)

Run from inside figures/: python figureS_vv.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import fig_style

# figure-specific overrides
plt.rcParams.update({'axes.titlesize': 8})

PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)
OUT_PNG = PLOT_DIR / 'figureS_vv.png'

# --- Physical constants (SI) ---
e_C       = 1.602176634e-19
eps0      = 8.8541878128e-12
J_per_eV  = 1.602176634e-19
A_to_m    = 1e-10
kB_eV     = 8.617333e-5

# --- Inputs ---
g_EF_mean   = 45.0
g_EF_low    = 20.0
V_cell_A3   = 4078.0
r_min_A     = 15.4
N_fu_per_cell = 72
V_fu_A3 = V_cell_A3 / N_fu_per_cell
delta_red_low  = 0.03
delta_red_high = 0.07
T_red_K = 1623
kBT_eV  = kB_eV * T_red_K

# --- Derived quantities ---
def lambda_TF(g_sup, V_A3):
    V_m3 = V_A3 * A_to_m**3
    g_vol = g_sup / J_per_eV / V_m3
    return np.sqrt(eps0 / (e_C**2 * g_vol))

lam_mean = lambda_TF(g_EF_mean, V_cell_A3)
lam_low  = lambda_TF(g_EF_low,  V_cell_A3)

def r_WS(delta):
    """Wigner-Seitz radius for vacancies on the O sublattice at non-stoichiometry delta."""
    return (V_fu_A3 / (4 * np.pi * delta))**(1/3)

r_WS_high_delta = r_WS(delta_red_high)
r_WS_low_delta  = r_WS(delta_red_low)

# --- Curves ---
r_grid_A = np.linspace(2.0, 20.0, 2000)
r_grid_m = r_grid_A * A_to_m
V_bare    = (2*e_C)**2 / (4*np.pi*eps0*r_grid_m) / J_per_eV
V_TF_mean = V_bare * np.exp(-r_grid_m / lam_mean)
V_TF_low  = V_bare * np.exp(-r_grid_m / lam_low)

fig = plt.figure(figsize=(fig_style.cm2in(fig_style.EES_SINGLE_CM),
                          fig_style.cm2in(7.5)))
ax = fig.add_axes([0.165, 0.17, 0.815, 0.66])

# Experimental delta range as shaded vertical band
ax.axvspan(r_WS_high_delta, r_WS_low_delta, color='#e8d8c0',
           alpha=0.55, lw=0, zorder=0.3)
ax.text(r_WS_high_delta + 1.0, 1.6,
        rf'$\delta_{{\mathrm{{red}}}} = {delta_red_low:.2f}-{delta_red_high:.2f}$',
        fontsize=6.5, color='#7a4e10', ha='left', va='top',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.2))

# k_BT horizontal line
ax.axhline(kBT_eV, color='#a06030', lw=0.8, ls=':', zorder=0.4)
ax.text(19.7, kBT_eV * 1.7,
        rf'$k_\mathrm{{B}}T$ at {T_red_K} K  ({kBT_eV*1000:.0f} meV)',
        fontsize=6.5, color='#a06030', ha='right', va='bottom')

# r_min light dashed marker (supercell side-reference)
ax.axvline(r_min_A, color='#aaaaaa', lw=0.5, ls='--', zorder=0.45)
ax.text(r_min_A - 0.2, 2e-11, r'$r_\mathrm{img}$',
        fontsize=6.5, color='#888888', va='center', ha='right',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.2))

# TF-screened curves
ax.plot(r_grid_A, V_TF_low,  color='#d97706', lw=1.2,
        label=rf'TF, $\lambda_\mathrm{{TF}} = {lam_low*1e10:.2f}$ Å (low $g$)')
ax.plot(r_grid_A, V_TF_mean, color='#1f5fb5', lw=1.2,
        label=rf'TF, $\lambda_\mathrm{{TF}} = {lam_mean*1e10:.2f}$ Å (mean $g$)')

ax.set_yscale('log')
ax.set_xlim(2, 20)
ax.set_ylim(1e-13, 5)
ax.set_xlabel(r'$r$  (${\AA}$)')
ax.set_ylabel(r'$V_{\mathrm{vv}}(r)$ for $q = 2e$  (eV)')

# Top secondary x-axis
def r_to_delta(r):
    return V_fu_A3 / (4 * np.pi * np.maximum(r, 1e-6)**3)
def delta_to_r(d):
    return (V_fu_A3 / (4 * np.pi * np.maximum(d, 1e-9)))**(1/3)

ax_top = ax.secondary_xaxis('top', functions=(r_to_delta, delta_to_r))
ax_top.set_xlabel(r'$\delta$ (random distribution, $r = r_\mathrm{WS}$)',
                  fontsize=8)
ax_top.tick_params(labelsize=7)
delta_ticks = [0.05, 0.02, 0.01, 0.005, 0.002]
ax_top.set_xticks(delta_ticks)
ax_top.set_xticklabels([f'{d:g}' for d in delta_ticks])

ax.legend(loc='lower left', fontsize=6.5, frameon=True,
          facecolor='white', edgecolor='none', framealpha=0.85,
          handletextpad=0.4, labelspacing=0.25,
          bbox_to_anchor=(0.02, 0.02))

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)

# Numbers for the prose
print("Key numbers for the prose:")
for d in [delta_red_low, delta_red_high]:
    r_A = r_WS(d)
    V_m = ((2*e_C)**2 / (4*np.pi*eps0*r_A*A_to_m) / J_per_eV) * np.exp(-r_A*A_to_m/lam_mean)
    V_l = ((2*e_C)**2 / (4*np.pi*eps0*r_A*A_to_m) / J_per_eV) * np.exp(-r_A*A_to_m/lam_low)
    print(f"  delta = {d:.2f}  ->  r_WS = {r_A:.2f} A  "
          f"V_TF(mean) = {V_m*1000:.2f} meV  V_TF(low) = {V_l*1000:.1f} meV")
V_rmin = ((2*e_C)**2 / (4*np.pi*eps0*r_min_A*A_to_m) / J_per_eV) * np.exp(-r_min_A*A_to_m/lam_mean)
print(f"  r_min = {r_min_A:.1f} A  ->  V_TF(mean) = {V_rmin*1e9:.2f} neV")

print(f"Wrote {OUT_PNG}")
