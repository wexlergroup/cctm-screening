"""Powder XRD of two CCTM compositions as synthesized and after cycling.

  Ca0.71Ce0.29Ti0.42Mn0.58O3 (CCTM1771014): as synthesized and after the
      four LSFR cycles of main-text Fig. 6.
  Ca0.67Ce0.33Ti0.33Mn0.67O3 (CCTM2112): as synthesized and after 20 TGA
      cycles (Protocols A, B, and C, then further cycling).

Each trace is normalized to its own maximum and offset. Composition colors
follow figureS_xrd.py (viridis by Mn content); the as-synthesized trace of
each pair is gray. The dashed line marks the CeO2(111) position.

Reads:  data/raw/xrd/CCTM1771014_fresh.xy, CCTM1771014_postLSFR.xy,
        CCTM2112_fresh.xy, CCTM2112_postTGA.xy (ASU exports, 28 Oct 2026)
Writes: plots/figureS_xrd_postcycling.png (600 dpi, EES single column)

Run from inside figures/: python figureS_xrd_postcycling.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import fig_style

RAW_DIR = Path('data') / 'raw' / 'xrd'
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_xrd_postcycling.png'

TWO_THETA_RANGE = (20, 80)
CEO2_111 = 28.55             # deg, Cu K-alpha1
OFFSET = 1.05                # vertical separation between traces
FRESH_COLOR = '0.45'

cmap = plt.cm.viridis
# (as-synthesized file, after-cycling file, composition label,
#  after-cycling label, composition color as in figureS_xrd.py)
PAIRS = [
    ('CCTM1771014_fresh.xy', 'CCTM1771014_postLSFR.xy',
     r'Ca$_{0.71}$Ce$_{0.29}$Ti$_{0.42}$Mn$_{0.58}$O$_3$',
     'after 4 LSFR cycles', cmap(0.50)),
    ('CCTM2112_fresh.xy', 'CCTM2112_postTGA.xy',
     r'Ca$_{0.67}$Ce$_{0.33}$Ti$_{0.33}$Mn$_{0.67}$O$_3$',
     'after 20 TGA cycles', cmap(0.85)),
]


def load(fn):
    d = np.loadtxt(RAW_DIR / fn)
    m = (d[:, 0] >= TWO_THETA_RANGE[0]) & (d[:, 0] <= TWO_THETA_RANGE[1])
    two_theta, intensity = d[m, 0], d[m, 1]
    return two_theta, intensity / intensity.max()


W = fig_style.cm2in(fig_style.EES_SINGLE_CM)
fig, ax = plt.subplots(figsize=(W, W * 1.0), dpi=600)

level = 0
for fresh, post, label, post_label, color in PAIRS:
    for fn, c, tag in ((fresh, FRESH_COLOR, 'as synthesized'),
                       (post, color, post_label)):
        two_theta, y = load(fn)
        ax.plot(two_theta, y + level * OFFSET, color=c, lw=0.6)
        ax.text(TWO_THETA_RANGE[1] - 0.5, level * OFFSET + 0.85, tag,
                ha='right', va='top', fontsize=6.5, color=c)
        level += 1
    ax.text(TWO_THETA_RANGE[1] - 0.5, (level - 1) * OFFSET + 0.62, label,
            ha='right', va='top', fontsize=6.5, color=color)

ax.axvline(CEO2_111, color='0.65', lw=0.5, ls=(0, (3, 2)), zorder=0)
ax.text(CEO2_111 - 0.6, level * OFFSET - 0.02, 'CeO$_2$ (111)',
        ha='right', va='top', fontsize=6.5, color='0.40')

ax.set_xlim(*TWO_THETA_RANGE)
ax.set_ylim(-0.05, level * OFFSET + 0.05)
ax.set_xlabel(r'$2\theta$ (deg, Cu $\mathrm{K}\alpha$)')
ax.set_ylabel('Intensity (normalized, offset)')
ax.set_yticks([])
for spine in ('left', 'right', 'top'):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'Wrote {OUT_PNG}')
