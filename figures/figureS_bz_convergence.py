"""Brillouin-zone sampling convergence check for CCTM3122 (ESI figure).

Single panel: spin-resolved total DOS for Ca0.75Ce0.25Ti0.50Mn0.50O3 at
Gamma-only and at a Gamma-centred 2x2x2 mesh, evaluated as single-point SCF
on the same Gamma-relaxed geometry. The total-energy difference between the
two grids is annotated.

Reads:
    data/dft_dos.csv (tidy long-form DOS table)

Writes:
    plots/figureS_bz_convergence.png (600 dpi, EES double-column 17.1 cm)

Run from inside figures/: python3 figureS_bz_convergence.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)
OUT_PNG = PLOT_DIR / 'figureS_bz_convergence.png'

E_WINDOW = (-8.0, 5.0)
ENERGY_GRID = np.linspace(E_WINDOW[0], E_WINDOW[1], 1000)

# Gaussian smoothing width in eV; converted to grid points using the dense
# ENERGY_GRID spacing so the kernel stays at 0.15 eV regardless of grid density.
SIGMA_EV = 0.15
SIGMA_PTS = SIGMA_EV / (ENERGY_GRID[1] - ENERGY_GRID[0])


def interp_and_smooth(e_native, d_native):
    """Linearly interpolate onto ENERGY_GRID, then apply a Gaussian kernel.

    The native DOS bin spacing is ~0.22 eV; convolving with a 0.15 eV
    Gaussian softens polyline kinks at the native bins without erasing
    band-edge or Fermi-level features.
    """
    dense = np.interp(ENERGY_GRID, e_native, d_native, left=0.0, right=0.0)
    return gaussian_filter1d(dense, SIGMA_PTS)


# figure-specific overrides
plt.rcParams.update({
    'axes.titlesize': 8,
})

df = pd.read_csv(DATA_DIR / 'dft_dos.csv')

sub_g = df[(df['comp_label'] == 'M0_N18') & (df['kgrid'] == 'gamma')]
sub_g = sub_g.sort_values('e_rel_EF').reset_index(drop=True)
sub_k = df[(df['comp_label'] == 'M0_N18') & (df['kgrid'] == '2x2x2')]
sub_k = sub_k.sort_values('e_rel_EF').reset_index(drop=True)

# Interpolate native DOS onto the dense ENERGY_GRID.
e_g = sub_g['e_rel_EF'].values
e_k = sub_k['e_rel_EF'].values
g_up = interp_and_smooth(e_g, sub_g['tot_up'].values)
g_dn = interp_and_smooth(e_g, sub_g['tot_dn'].values)
k_up = interp_and_smooth(e_k, sub_k['tot_up'].values)
k_dn = interp_and_smooth(e_k, sub_k['tot_dn'].values)

fig = plt.figure(figsize=(fig_style.cm2in(fig_style.EES_DOUBLE_CM),
                          fig_style.cm2in(6.5)))
ax = fig.add_axes([0.085, 0.19, 0.89, 0.78])

ax.plot(ENERGY_GRID,  g_up, color='#1f5fb5', lw=1.0, label=r'$\Gamma$-only')
ax.plot(ENERGY_GRID, -g_dn, color='#1f5fb5', lw=1.0)
ax.plot(ENERGY_GRID,  k_up, color='#d97706', lw=1.0, label=r'$2 \times 2 \times 2$')
ax.plot(ENERGY_GRID, -k_dn, color='#d97706', lw=1.0)
ax.axvline(0, color='black', lw=0.5, ls='--', alpha=0.6, zorder=0.5)
ax.axhline(0, color='#666666', lw=0.5, zorder=0.5)
ax.set_xlim(E_WINDOW)
ax.set_xlabel(r'$E - E_\mathrm{F}$ (eV)')
ax.set_ylabel('Total DOS (states/eV)')
ax.text(0.99, 0.96, r'Ca$_{0.75}$Ce$_{0.25}$Ti$_{0.50}$Mn$_{0.50}$O$_3$',
        transform=ax.transAxes, fontsize=7, va='top', ha='right',
        color='#222222')
ax.text(0.99, 0.04,
        r'$\Delta E_\mathrm{tot}$ = 14 meV/cell  (0.04 meV/atom)',
        transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
        color='#222222',
        bbox=dict(facecolor='white', alpha=0.85,
                  edgecolor='none', pad=1.2))
ax.legend(loc='lower left', fontsize=7,
          handletextpad=0.4, labelspacing=0.25,
          bbox_to_anchor=(0.015, 0.02),
          frameon=True, facecolor='white', edgecolor='none',
          framealpha=0.85)
# Spin labels in the left half, away from the right-side annotations
ax.text(0.05, 0.92, r'spin $\uparrow$', transform=ax.transAxes,
        fontsize=7, va='top', ha='left', color='#555555',
        bbox=dict(facecolor='white', alpha=0.85,
                  edgecolor='none', pad=1.2))
ax.text(0.05, 0.25, r'spin $\downarrow$', transform=ax.transAxes,
        fontsize=7, va='bottom', ha='left', color='#555555',
        bbox=dict(facecolor='white', alpha=0.85,
                  edgecolor='none', pad=1.2))

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Wrote {OUT_PNG}")
