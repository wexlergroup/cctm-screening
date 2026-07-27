"""Density of states across the CCTM bulk compositions (ESI figure).

Panel (a): spin-resolved total DOS averaged across all six bulk compositions
           (mean +- 1 standard deviation envelope), demonstrating finite DOS
           at the Fermi level (metallic) and full spin polarization at E_F
           (half-metallic in character) as a family-wide feature.
Panel (b): element-projected DOS (Ce 4f, Mn 3d, O 2p) for CCTM3122, showing
           that the states crossing E_F are predominantly Ce 4f character.

All curves are linearly interpolated onto a dense 1000-point energy grid
spanning the plot window and then Gaussian-smoothed with sigma = 0.15 eV.

Reads:
    data/dft_dos.csv (tidy long-form DOS table)

Writes:
    plots/figureS_dos.png (600 dpi, EES double-column 17.1 cm)

Run from inside figures/: python3 figureS_dos.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)
OUT_PNG = PLOT_DIR / 'figureS_dos.png'

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


EL_COLOURS = {
    'Ce_f': '#2e8b57',  # green
    'Mn_d': '#7a4ea1',  # purple
    'O_p':  '#c43a3a',  # red
}

# figure-specific overrides
plt.rcParams.update({
    'axes.titlesize': 8,
})

df = pd.read_csv(DATA_DIR / 'dft_dos.csv')

# Distinct compositions (gamma-grid only)
comp_labels = (
    df[df['kgrid'] == 'gamma'][['comp_label', 'x_Ce', 'x_Mn']]
    .drop_duplicates()
    .sort_values(['x_Ce', 'x_Mn'])
    ['comp_label']
    .tolist()
)
n_comp = len(comp_labels)


def formula_to_label(formula_str):
    out = ''
    i = 0
    while i < len(formula_str):
        ch = formula_str[i]
        if ch.isalpha():
            j = i + 1
            while j < len(formula_str) and formula_str[j].isalpha() and formula_str[j].islower():
                j += 1
            elem = formula_str[i:j]
            out += elem
            i = j
            k = i
            while k < len(formula_str) and (formula_str[k].isdigit() or formula_str[k] == '.'):
                k += 1
            if k > i:
                num = formula_str[i:k]
                out += '$_{3}$' if (num == '3' and elem == 'O') else f'$_{{{num}}}$'
                i = k
        else:
            out += ch
            i += 1
    return out


# ---------- Figure ----------
fig = plt.figure(figsize=(fig_style.cm2in(fig_style.EES_DOUBLE_CM),
                          fig_style.cm2in(9.5)))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0],
                      hspace=0.50, left=0.085, right=0.97,
                      top=0.98, bottom=0.085)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# === Panel (a): composition-averaged total DOS ===
up_stack = np.zeros((n_comp, len(ENERGY_GRID)))
dn_stack = np.zeros((n_comp, len(ENERGY_GRID)))
for ic, label in enumerate(comp_labels):
    sub = df[(df['comp_label'] == label) & (df['kgrid'] == 'gamma')]
    sub = sub.sort_values('e_rel_EF')
    e_n = sub['e_rel_EF'].values
    up_stack[ic] = interp_and_smooth(e_n, sub['tot_up'].values)
    dn_stack[ic] = interp_and_smooth(e_n, sub['tot_dn'].values)

mean_up = up_stack.mean(axis=0)
std_up  = up_stack.std(axis=0)
mean_dn = dn_stack.mean(axis=0)
std_dn  = dn_stack.std(axis=0)

ENV_COLOR  = '#4a5d8a'
MEAN_COLOR = '#1f2d4d'

ax_a.fill_between(ENERGY_GRID,  mean_up - std_up,  mean_up + std_up,
                  color=ENV_COLOR, alpha=0.25, lw=0, zorder=2)
ax_a.fill_between(ENERGY_GRID, -(mean_dn - std_dn), -(mean_dn + std_dn),
                  color=ENV_COLOR, alpha=0.25, lw=0, zorder=2)
ax_a.plot(ENERGY_GRID,  mean_up, color=MEAN_COLOR, lw=1.1, zorder=3)
ax_a.plot(ENERGY_GRID, -mean_dn, color=MEAN_COLOR, lw=1.1, zorder=3)

ax_a.axvline(0, color='black', lw=0.5, ls='--', alpha=0.6, zorder=0.5)
ax_a.axhline(0, color='#666666', lw=0.5, zorder=0.5)
ax_a.set_xlim(E_WINDOW)
ax_a.set_xlabel(r'$E - E_\mathrm{F}$ (eV)')
ax_a.set_ylabel('Total DOS (states/eV)')
ax_a.text(0.012, 0.95, '(a)', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top', ha='left')
ax_a.text(0.985, 0.92, r'spin $\uparrow$', transform=ax_a.transAxes,
          fontsize=7, va='top', ha='right', color='#555555')
ax_a.text(0.985, 0.08, r'spin $\downarrow$', transform=ax_a.transAxes,
          fontsize=7, va='bottom', ha='right', color='#555555')

mean_handle = plt.Line2D([0], [0], color=MEAN_COLOR, lw=1.1,
                         label='mean over 6 compositions')
env_handle  = Patch(facecolor=ENV_COLOR, alpha=0.25, label=r'$\pm 1\sigma$')
ax_a.legend(handles=[env_handle, mean_handle],
            loc='lower left', fontsize=7,
            handletextpad=0.4, labelspacing=0.25,
            bbox_to_anchor=(0.015, 0.02),
            frameon=True, facecolor='white', edgecolor='none',
            framealpha=0.85)

# === Panel (b): orbital projection for CCTM3122 ===
sub_b = df[(df['comp_label'] == 'M0_N18') & (df['kgrid'] == 'gamma')]
sub_b = sub_b.sort_values('e_rel_EF')
e_b = sub_b['e_rel_EF'].values

tot_up_d  = interp_and_smooth(e_b, sub_b['tot_up'].values)
tot_dn_d  = interp_and_smooth(e_b, sub_b['tot_dn'].values)
ce_f_up_d = interp_and_smooth(e_b, sub_b['Ce_f_up'].values)
ce_f_dn_d = interp_and_smooth(e_b, sub_b['Ce_f_dn'].values)
mn_d_up_d = interp_and_smooth(e_b, sub_b['Mn_d_up'].values)
mn_d_dn_d = interp_and_smooth(e_b, sub_b['Mn_d_dn'].values)
o_p_up_d  = interp_and_smooth(e_b, sub_b['O_p_up'].values)
o_p_dn_d  = interp_and_smooth(e_b, sub_b['O_p_dn'].values)

ax_b.fill_between(ENERGY_GRID,  tot_up_d, 0,
                  color='#cccccc', alpha=0.5, lw=0, zorder=1)
ax_b.fill_between(ENERGY_GRID, -tot_dn_d, 0,
                  color='#cccccc', alpha=0.5, lw=0, zorder=1)
ax_b.plot(ENERGY_GRID,  ce_f_up_d, color=EL_COLOURS['Ce_f'], lw=1.0,
          zorder=3, label='Ce 4f')
ax_b.plot(ENERGY_GRID, -ce_f_dn_d, color=EL_COLOURS['Ce_f'], lw=1.0, zorder=3)
ax_b.plot(ENERGY_GRID,  mn_d_up_d, color=EL_COLOURS['Mn_d'], lw=1.0,
          zorder=3, label='Mn 3d')
ax_b.plot(ENERGY_GRID, -mn_d_dn_d, color=EL_COLOURS['Mn_d'], lw=1.0, zorder=3)
ax_b.plot(ENERGY_GRID,  o_p_up_d,  color=EL_COLOURS['O_p'], lw=1.0,
          zorder=3, label='O 2p')
ax_b.plot(ENERGY_GRID, -o_p_dn_d,  color=EL_COLOURS['O_p'], lw=1.0, zorder=3)
total_handle = Patch(facecolor='#cccccc', alpha=0.5, label='total')

ax_b.axvline(0, color='black', lw=0.5, ls='--', alpha=0.6, zorder=0.5)
ax_b.axhline(0, color='#666666', lw=0.5, zorder=0.5)
ax_b.set_xlim(E_WINDOW)
ax_b.set_xlabel(r'$E - E_\mathrm{F}$ (eV)')
ax_b.set_ylabel('DOS (states/eV)')
ax_b.text(0.012, 0.95, '(b)', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top', ha='left')
ax_b.text(0.985, 0.95, formula_to_label('Ca0.75Ce0.25Ti0.50Mn0.50O3'),
          transform=ax_b.transAxes, fontsize=7, va='top', ha='right',
          color='#222222')
handles_b, labels_b = ax_b.get_legend_handles_labels()
handles_b = [total_handle] + handles_b
labels_b  = ['total'] + labels_b
ax_b.legend(handles_b, labels_b, loc='lower left', fontsize=7,
            frameon=True, facecolor='white', edgecolor='none', framealpha=0.85,
            handletextpad=0.4, labelspacing=0.25,
            bbox_to_anchor=(0.012, 0.02))

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Wrote {OUT_PNG}")
