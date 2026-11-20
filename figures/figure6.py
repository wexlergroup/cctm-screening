"""Fig. 6: LSFR cycling data for two CCTM compositions.

Each column shows one composition: O2 production rate during
laser-driven reduction (red) and H2 production rate during water-splitting
reoxidation (blue), both in mmol per (mol of perovskite atoms) per second.
Shaded bands mark the intervals in which the reactor's laser flag
('Laser' = 1 in the O2 analysis windows) is on, including the heating and
cooling ramps. No temperature panel is drawn: the furnace thermocouple
(850 deg C reoxidation) does not see the laser-heated sample, and the
pyrometer read ~20 deg C high under steam, so both temperatures are stated
in the caption (revision decisions 2026-10-26 and, after N. Ali's comment
that a shaded thermocouple panel reads as a 900 deg C laser set point,
2026-11-03).
Cycle integration is performed by the LSFR's onboard analysis software;
the per-cycle integrated values (anmeta.area_mmolmols) are printed after
the figure is saved for cross-checking quoted numbers.

Reads: data/lsfr_CCTM1771014.json, data/lsfr_CCTM2112.json
Writes: plots/figure6.png
Run from inside figures/: python3 figure6.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')

OUT_PNG = PLOT_DIR / 'figure6.png'

RUNS = [
    ('lsfr_CCTM1771014.json',
     r'Ca$_{0.71}$Ce$_{0.29}$Ti$_{0.42}$Mn$_{0.58}$O$_3$'),
    ('lsfr_CCTM2112.json',
     r'Ca$_{0.67}$Ce$_{0.33}$Ti$_{0.33}$Mn$_{0.67}$O$_3$'),
]

# Color scheme: O2 (reduction product, hot) = deep red;
# H2 (oxidation product, cool) = deep blue.
O2_COLOR = '#c0392b'
H2_COLOR = '#1f6cb5'
LASER_ALPHA = 0.15   # laser-on shading, in the O2 (reduction) color


def laser_on_intervals(json_path):
    """Return (t_start, t_end) in minutes for each contiguous laser-on
    stretch, read from the 'Laser' flag of the O2 analysis windows."""
    with open(json_path) as f:
        d = json.load(f)
    spans = []
    for ad_key in sorted(k for k in d['05'].keys() if k.startswith('andat')):
        s = d['05'][ad_key]['set']
        if 'Laser' not in s:
            continue
        t = np.array(s['time']) / 60.0
        on = np.asarray(s['Laser'], dtype=float) > 0.5
        # rising and falling edges of the on-flag
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            spans.append((t[i0], t[i1 - 1]))
    return spans


def load_cycle05(json_path):
    """Return list of (species, t_minutes, rate, area, ad_key) tuples
    ordered by their original sequence in the data file."""
    with open(json_path) as f:
        d = json.load(f)
    out = []
    for ad_key in sorted(k for k in d['05'].keys() if k.startswith('andat')):
        ad = d['05'][ad_key]
        t = np.array(ad['set']['time']) / 60.0   # min
        species = 'O2' if 'O2_denoise_corr_mmolmols' in ad['set'] else 'H2'
        rate_key = f'{species}_denoise_corr_mmolmols'
        rate = np.array(ad['set'][rate_key])
        area = ad['anmeta']['area_mmolmols'][0]
        out.append((species, t, rate, area, ad_key))
    return out


# ---------- Build figure ----------
W = fig_style.cm2in(fig_style.EES_DOUBLE_CM)
fig, axes_rate = plt.subplots(1, 2, figsize=(W, W * 0.38), dpi=600,
                              gridspec_kw=dict(wspace=0.18))

cycle_counter_per_comp = []

for c, (fn, comp_label) in enumerate(RUNS):
    runs = load_cycle05(DATA_DIR / fn)
    ax_r = axes_rate[c]

    laser_spans = laser_on_intervals(DATA_DIR / fn)
    for t0, t1 in laser_spans:
        ax_r.axvspan(t0, t1, color=O2_COLOR, alpha=LASER_ALPHA, lw=0,
                     zorder=0)

    cycle_idx = 0       # increments on every O2 event

    for species, t, rate, area, ad_key in runs:
        color = O2_COLOR if species == 'O2' else H2_COLOR
        ax_r.plot(t, rate, color=color, lw=0.7, zorder=2)
        if species == 'O2':
            cycle_idx += 1

    # Cycle-number digits centered on each laser-on band, near the top of
    # the panel and drawn in front of the band. One band per O2 window, in
    # cycle order. The figure caption explains the numbering.
    CYCLE_LABEL_Y = 0.195
    assert len(laser_spans) == cycle_idx
    for cyc, (t0, t1) in enumerate(laser_spans, start=1):
        ax_r.text(0.5 * (t0 + t1), CYCLE_LABEL_Y, f'{cyc}', fontsize=7,
                  ha='center', va='center', color='0.25', zorder=3)

    cycle_counter_per_comp.append(cycle_idx)

    ax_r.set_title(comp_label, fontsize=8)
    ax_r.set_ylim(-0.005, 0.22)
    ax_r.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax_r.set_xlim(0, 130)
    ax_r.set_xticks([0, 30, 60, 90, 120])
    ax_r.set_xlabel('Time (min)')
    ax_r.spines['top'].set_visible(False)
    if c == 0:
        ax_r.set_ylabel('Rate (mmol mol$_\\mathrm{atom}^{-1}$ s$^{-1}$)',
                        fontsize=7.5)

# Top-of-figure legend with O2 / H2 traces
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Line2D([0], [0], color=O2_COLOR, lw=1.0,
           label=r'O$_2$ rate (reduction)'),
    Line2D([0], [0], color=H2_COLOR, lw=1.0,
           label=r'H$_2$ rate (reoxidation)'),
    Patch(facecolor=O2_COLOR, alpha=LASER_ALPHA, edgecolor='none',
          label='Laser on'),
]
fig.legend(handles=handles, loc='upper center',
           bbox_to_anchor=(0.5, 1.04), fontsize=7,
           frameon=False, ncol=3, handlelength=2.5)

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)

# Print integrated values for reference / table cross-check
print('Integrated O2 / H2 per cycle (mmol/(mol·atom)):')
for fn, comp_label in RUNS:
    print(f'  {comp_label}')
    for sp, t, rate, area, ad_key in load_cycle05(DATA_DIR / fn):
        print(f'    {ad_key}: {sp} = {area:.3f}')

print(f"Wrote {OUT_PNG}")
