"""Stacked-overlay XRD patterns of the synthesized CCTM samples with a
calculated reference pattern from the CCTM2112 SQS supercell.

The calculated reference (CCTM2112 SQS, 360-atom 3x3x2 supercell, P1) is
computed via pymatgen XRDCalculator with Cu Kalpha1 (1.54056 angstrom) and
convolved with a 0.1 deg FWHM Gaussian for visual comparison with the
experimental traces.

Reads:  data/xrd_CCTM3122.csv, data/xrd_CCTM1771014.csv, data/xrd_CCTM2112.csv
        data/cctm2112_sqs.cif (calculated-pattern structure)
Writes: plots/figureS_xrd.png (600 dpi, EES single column)

Run from inside figures/: python figureS_xrd.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_xrd.png'

# Experimental files, ordered by ascending X_Mn (this is also the
# reduction-favorability order along the composition trajectory).
EXPERIMENTAL = [
    ('xrd_CCTM3122.csv',
     r'Ca$_{0.75}$Ce$_{0.25}$Ti$_{0.50}$Mn$_{0.50}$O$_3$',
     0.50),
    ('xrd_CCTM1771014.csv',
     r'Ca$_{0.71}$Ce$_{0.29}$Ti$_{0.42}$Mn$_{0.58}$O$_3$',
     0.58),
    ('xrd_CCTM2112.csv',
     r'Ca$_{0.67}$Ce$_{0.33}$Ti$_{0.33}$Mn$_{0.67}$O$_3$',
     0.67),
]

CALC_CIF = DATA_DIR / 'cctm2112_sqs.cif'        # CCTM2112 SQS supercell
TWO_THETA_RANGE = (20, 80)
GAUSSIAN_FWHM = 0.10                            # deg; calc pattern smoothing


# ---------- Calculated reference pattern ----------
def calculate_reference_pattern(cif_path, two_theta_grid, fwhm=0.10):
    """Compute Cu K-alpha1 XRD pattern from CIF and convolve onto a
    common 2theta grid with a Gaussian of given FWHM (deg)."""
    structure = Structure.from_file(cif_path)
    xrd = XRDCalculator(wavelength='CuKa1')
    pattern = xrd.get_pattern(structure, two_theta_range=TWO_THETA_RANGE)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))     # deg
    intensity = np.zeros_like(two_theta_grid, dtype=float)
    for tt, I in zip(pattern.x, pattern.y):
        intensity += I * np.exp(-0.5 * ((two_theta_grid - tt) / sigma) ** 2)
    return intensity / intensity.max()


# ---------- Plot ----------
def render(experimental_subset, out_png):
    n = len(experimental_subset)
    OFFSET = 1.05            # vertical separation between traces
    CALC_OFFSET = -OFFSET    # calculated pattern below baseline

    W = fig_style.cm2in(fig_style.EES_SINGLE_CM)
    height = W * (1.0 if n == 4 else 0.88)
    fig, ax = plt.subplots(figsize=(W, height), dpi=600)

    cmap = plt.cm.viridis
    if n == 1:
        colors = [cmap(0.45)]
    else:
        colors = [cmap(0.15 + 0.7 * i / (n - 1)) for i in range(n)]

    # Build a common 2theta grid from the first experimental file.
    first = np.loadtxt(DATA_DIR / experimental_subset[0][0], delimiter=',', skiprows=1)
    two_theta_grid = first[(first[:, 0] >= TWO_THETA_RANGE[0]) &
                           (first[:, 0] <= TWO_THETA_RANGE[1]), 0]

    # Calculated reference at the bottom.
    I_calc = calculate_reference_pattern(CALC_CIF, two_theta_grid,
                                         fwhm=GAUSSIAN_FWHM)
    ax.plot(two_theta_grid, I_calc + CALC_OFFSET, color='black', lw=0.6)
    ax.text(TWO_THETA_RANGE[1] - 0.5, CALC_OFFSET + 0.85,
            r'Ca$_{0.67}$Ce$_{0.33}$Ti$_{0.33}$Mn$_{0.67}$O$_3$ (calculated)',
            ha='right', va='top', fontsize=6.5, color='black')

    # Experimental traces.
    for i, (fn, label, xMn) in enumerate(experimental_subset):
        d = np.loadtxt(DATA_DIR / fn, delimiter=',', skiprows=1)
        m = (d[:, 0] >= TWO_THETA_RANGE[0]) & (d[:, 0] <= TWO_THETA_RANGE[1])
        two_theta, I = d[m, 0], d[m, 1]
        I_norm = I / I.max()
        ax.plot(two_theta, I_norm + i * OFFSET, color=colors[i], lw=0.6)
        ax.text(TWO_THETA_RANGE[1] - 0.5, i * OFFSET + 0.85, label,
                ha='right', va='top', fontsize=6.5, color=colors[i])

    # Mark the absent CeO2 (111) reflection (2theta = 28.55 deg, Cu K-alpha1).
    y_top = (n - 1) * OFFSET + 1.0
    ax.axvline(28.55, color='0.65', lw=0.5, ls=(0, (3, 2)), zorder=0)
    # Label sits in the clean baseline region between the calculated trace
    # and the lowest experimental trace, where the dashed line passes through
    # empty space.
    ax.text(28.55, -0.55, 'CeO$_2$\n(111)',
            ha='center', va='center', fontsize=6.5, color='0.40',
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='none', alpha=0.9))

    ax.set_xlim(*TWO_THETA_RANGE)
    ax.set_ylim(CALC_OFFSET - 0.05, y_top + 0.20)
    ax.set_xlabel(r'$2\theta$ (deg, Cu $\mathrm{K}\alpha$)')
    ax.set_ylabel('Intensity (normalized, offset)')
    ax.set_yticks([])
    for spine in ('left', 'right', 'top'):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close(fig)


render(EXPERIMENTAL, OUT_PNG)
print(f"Wrote {OUT_PNG}")
