"""Fig. 6: LSFR cycling data for two CCTM compositions.

Each column shows one composition. Top panel: O2 production rate during
laser-driven reduction (red) and H2 production rate during water-splitting
reoxidation (blue), both in mmol per (mol of perovskite atoms) per second.
Bottom panel: sample-surface temperature (pyrometer) and furnace
temperature, showing the laser-driven cycling between 850 and 1350 deg C.
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
from matplotlib.gridspec import GridSpec

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
# H2 (oxidation product, cool) = deep blue. Temperature in gray.
O2_COLOR = '#c0392b'
H2_COLOR = '#1f6cb5'
T_COLOR = '0.35'


def load_cycle05(json_path):
    """Return list of (species, t_minutes, rate, T_sample_or_FT, area) tuples
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
        # Temperature: prefer PY TEMP (sample surface) when available
        T = np.array(ad['set']['PY TEMP'] if 'PY TEMP' in ad['set']
                     else ad['set']['FT TEMP'])
        area = ad['anmeta']['area_mmolmols'][0]
        out.append((species, t, rate, T, area, ad_key))
    return out


# ---------- Build figure ----------
W = fig_style.cm2in(fig_style.EES_DOUBLE_CM)
fig = plt.figure(figsize=(W, W * 0.50), dpi=600)
gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.10, wspace=0.18,
              figure=fig)

axes_rate = [fig.add_subplot(gs[0, c]) for c in range(2)]
axes_T = [fig.add_subplot(gs[1, c], sharex=axes_rate[c]) for c in range(2)]

cycle_counter_per_comp = []

for c, (fn, comp_label) in enumerate(RUNS):
    runs = load_cycle05(DATA_DIR / fn)
    ax_r = axes_rate[c]
    ax_T = axes_T[c]

    cycle_idx = 0       # increments on every O2 event
    annotations = []

    for species, t, rate, T, area, ad_key in runs:
        color = O2_COLOR if species == 'O2' else H2_COLOR
        ax_r.plot(t, rate, color=color, lw=0.7)
        ax_T.plot(t, T, color=T_COLOR, lw=0.5)

        if species == 'O2':
            cycle_idx += 1
            # Annotate cycle number + integrated O2 above the peak
            i_max = int(np.argmax(rate))
            annotations.append((t[i_max], rate[i_max], area, cycle_idx, 'O2'))
        else:
            i_max = int(np.argmax(rate))
            annotations.append((t[i_max], rate[i_max], area,
                                cycle_idx, 'H2'))

    # Cycle-number digits above each O2 peak, on a uniform row near the top
    # of the panel. The figure caption explains the numbering.
    CYCLE_LABEL_Y = 0.195
    for t_peak, r_peak, area, cyc, sp in annotations:
        if sp == 'O2':
            ax_r.text(t_peak, CYCLE_LABEL_Y,
                      f'{cyc}', fontsize=7,
                      ha='center', va='center', color='0.25')

    cycle_counter_per_comp.append(cycle_idx)

    ax_r.set_title(comp_label, fontsize=8)
    ax_r.set_ylim(-0.005, 0.22)
    ax_r.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax_r.tick_params(axis='x', labelbottom=False)
    ax_r.spines['top'].set_visible(False)
    if c == 0:
        ax_r.set_ylabel('Rate (mmol mol$_\\mathrm{atom}^{-1}$ s$^{-1}$)',
                        fontsize=7.5)

    ax_T.set_ylim(800, 1400)
    ax_T.set_yticks([850, 1350])
    ax_T.set_xlim(0, 130)
    ax_T.set_xticks([0, 30, 60, 90, 120])
    ax_T.set_xlabel('Time (min)')
    ax_T.spines['top'].set_visible(False)
    if c == 0:
        ax_T.set_ylabel('T (°C)', fontsize=7.5)
    if c == 1:
        # hide y-axis tick labels on right panel of T axis to reduce repetition
        ax_T.tick_params(axis='y', labelleft=True)

# Top-of-figure legend with O2 / H2 traces
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=O2_COLOR, lw=1.0,
           label=r'O$_2$ rate (reduction)'),
    Line2D([0], [0], color=H2_COLOR, lw=1.0,
           label=r'H$_2$ rate (re-oxidation)'),
]
fig.legend(handles=handles, loc='upper center',
           bbox_to_anchor=(0.5, 1.02), fontsize=7,
           frameon=False, ncol=2, handlelength=2.5)

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)

# Print integrated values for reference / table cross-check
print('Integrated O2 / H2 per cycle (mmol/(mol·atom)):')
for fn, comp_label in RUNS:
    print(f'  {comp_label}')
    for sp, t, rate, T, area, ad_key in load_cycle05(DATA_DIR / fn):
        print(f'    {ad_key}: {sp} = {area:.3f}')

print(f"Wrote {OUT_PNG}")
