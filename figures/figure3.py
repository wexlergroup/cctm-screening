"""Figure 3: oxygen-vacancy formation energy distributions across the sampled bulks.

For each of the six bulk compositions, the 15 symmetry-distinct local 1NN
cells (combinations of x_Ce in {0, 1/4, 1/2, 3/4, 1} and x_Mn in {0, 1/2, 1})
carry a vacancy count and a per-cell mean E_v. Each violin shows the
Count-weighted distribution of E_v across local environments, i.e. what a
randomly selected vacancy in that bulk would experience; small dots show
the 15 individual cell means. The shaded band marks the target E_v window
(3.4-3.9 eV) for solar thermochemical H2 production.

Reads:
    data/dft_ev_local_env.csv
Writes:
    plots/figure3.png
Run from inside figures/: python3 figure3.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')

DATA_CSV = DATA_DIR / 'dft_ev_local_env.csv'
OUT_PNG = PLOT_DIR / 'figure3.png'
TARGET_EV = (3.4, 3.9)


# ---------- Load and classify ----------
df = pd.read_csv(DATA_CSV)


def classify(row, atol=1e-3):
    on_A = abs(row['XMn'] - 2 * row['XCe']) < atol
    on_B = abs(row['XCe'] - 1 / 3) < atol
    if on_A and on_B:
        return 'intersection'
    if on_A:
        return 'series_A'
    if on_B:
        return 'series_B'
    return 'other'


df['group'] = df.apply(classify, axis=1)

# Drop columns not used by the figure
keep = ['Composition', 'M', 'N', 'XCe', 'XMn',
        'xCe', 'xMn', 'Count', 'Ev', 'group']
df = df[keep].copy()


# ---------- Order the bulks for plotting ----------
# Series A (N ascending), then Series B (M = 6, M = -6)
bulk_order = [
    ('M=0, N=18',  'series_A'),
    ('M=0, N=21',  'series_A'),
    ('M=0, N=24',  'intersection'),
    ('M=0, N=27',  'series_A'),
    ('M=6, N=24',  'series_B'),
    ('M=-6, N=24', 'series_B'),
]
xtick_labels = [
    '(0.25, 0.50)',  # M=0,  N=18
    '(0.29, 0.58)',  # M=0,  N=21
    '(0.33, 0.67)',  # M=0,  N=24  (CCTM2112)
    '(0.38, 0.75)',  # M=0,  N=27
    '(0.33, 0.58)',  # M=6,  N=24
    '(0.33, 0.75)',  # M=-6, N=24
]


# ---------- Plot ----------
W = fig_style.cm2in(fig_style.EES_SINGLE_CM)
fig, ax = plt.subplots(figsize=(W, W * 1.05), dpi=600)

c_A, c_B, c_X = '#1f6cb5', '#c0392b', '#e8b923'
group_color = {'series_A': c_A, 'series_B': c_B, 'intersection': c_X}

# Target E_v band (drawn first so violins sit on top)
ax.axhspan(*TARGET_EV, color='0.92', zorder=0)

positions = list(range(1, 7))
for pos, (comp_name, grp) in zip(positions, bulk_order):
    sub = df[df['Composition'] == comp_name]

    # Count-weighted samples for the violin KDE
    samples = np.repeat(sub['Ev'].values, sub['Count'].values.astype(int))

    parts = ax.violinplot(samples, positions=[pos], widths=0.78,
                          showmeans=False, showmedians=True, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(group_color[grp])
        body.set_edgecolor('black')
        body.set_alpha(0.55)
        body.set_linewidth(0.5)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(0.8)

    # Overlay individual cell means (uniform size)
    ax.scatter(np.full(len(sub), pos), sub['Ev'].values,
               s=8, c='black', alpha=0.7, zorder=3, linewidths=0)

# Axis cosmetics
ax.set_xticks(positions)
ax.set_xticklabels(xtick_labels, rotation=30, ha='right',
                   rotation_mode='anchor')
ax.set_xlabel(r'$(X_\mathrm{Ce},\;X_\mathrm{Mn})$')
ax.set_ylabel(r'$E_\mathrm{v}$ (eV)')
ax.set_xlim(0.4, 6.6)
ax.set_ylim(2.45, 4.95)

# Target-band label, placed in the right margin outside the axes
ax.text(1.012, 0.5 * (TARGET_EV[0] + TARGET_EV[1]),
        'target', ha='left', va='center',
        fontsize=6.5, color='0.35',
        transform=ax.get_yaxis_transform(),
        clip_on=False)

# Legend below the axes, single horizontal row
leg_handles = [
    Patch(facecolor=c_A, edgecolor='black', alpha=0.55,
          label=r'$X_\mathrm{Mn} = 2\,X_\mathrm{Ce}$'),
    Patch(facecolor=c_X, edgecolor='black', alpha=0.55,
          label='CCTM2112'),
    Patch(facecolor=c_B, edgecolor='black', alpha=0.55,
          label=r'$X_\mathrm{Ce} = 1/3$'),
]
fig.tight_layout()
fig.subplots_adjust(bottom=0.32)

# Place the legend so the white space between its top and the bottom of the
# x-axis label equals the space between the top of the x-axis label and the
# bottom of the x-tick labels (measured from the drawn canvas).
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
xlab_bb = ax.xaxis.label.get_window_extent(renderer)
ticks_y0 = min(t.get_window_extent(renderer).y0
               for t in ax.get_xticklabels())
gap_px = ticks_y0 - xlab_bb.y1
leg_top_fig = (xlab_bb.y0 - gap_px) / fig.bbox.height
leg = fig.legend(handles=leg_handles,
                 loc='upper center', bbox_to_anchor=(0.5, leg_top_fig),
                 ncol=3, frameon=False, fontsize=6.5,
                 handletextpad=0.4, columnspacing=1.2)
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
print(f"Wrote {OUT_PNG}")
