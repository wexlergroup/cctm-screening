"""Site-resolved magnetic moments per element and composition (ESI figure).

Single panel: box-and-whisker distribution of |mu| by element across all six
bulk SQS supercells, with Mn^{4+}/^{3+}/^{2+} assignment band edges (3.4 and
4.0 uB) and the Ce^{4+}/Ce^{3+} cutoff (0.4 uB) drawn. Also prints the
PAW-vs-Bader validation numbers and the ESI table of Ce oxidation-state
counts per composition.

Reads:
    data/dft_magmoments_bulk.csv (per-site bulk moments, 2160 rows)

Writes:
    plots/figureS_magmoments.png (600 dpi, EES single-column 8.5 cm)

Run from inside figures/: python3 figureS_magmoments.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)
OUT_PNG = PLOT_DIR / 'figureS_magmoments.png'

CUTOFF_CE   = 0.4
MN43_BAND   = 3.4
MN32_BAND   = 4.0

EL_ORDER = ['Ca', 'Ce', 'Ti', 'Mn', 'O']

# ---------- Load data ----------
b = pd.read_csv(DATA_DIR / 'dft_magmoments_bulk.csv')

# Composition ordering for the legend (by x_Ce, then x_Mn)
comp_order = (
    b[['comp_label', 'formula', 'x_Ce', 'x_Mn']]
    .drop_duplicates()
    .sort_values(['x_Ce', 'x_Mn'])
    .reset_index(drop=True)
)
n_comp = len(comp_order)
viridis = plt.get_cmap('viridis')
comp_colours = {row['comp_label']: viridis(i / (n_comp - 1))
                for i, row in comp_order.iterrows()}

def formula_to_label(formula_str):
    """Ca0.75Ce0.25Ti0.50Mn0.50O3 -> matplotlib mathtext label."""
    s = formula_str
    out = ''
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isalpha():
            j = i + 1
            while j < len(s) and s[j].isalpha() and s[j].islower():
                j += 1
            elem = s[i:j]
            out += elem
            i = j
            k = i
            while k < len(s) and (s[k].isdigit() or s[k] == '.'):
                k += 1
            if k > i:
                num = s[i:k]
                if num == '3' and elem == 'O':
                    out += '$_{3}$'
                else:
                    out += f'$_{{{num}}}$'
                i = k
        else:
            out += ch
            i += 1
    return out

# ---------- Figure ----------
fig = plt.figure(figsize=(fig_style.cm2in(fig_style.EES_SINGLE_CM),
                          fig_style.cm2in(8.5)))
# Axes positioned to leave a compact bottom strip for the 6-entry legend
# in 2 cols x 3 rows, matching the top/side margin generosity.
ax = fig.add_axes([0.13, 0.265, 0.84, 0.715])

# === Boxplot of |mu| by element x composition ===
group_spacing = 0.40       # spacing between element groups
box_width     = 0.11       # per-composition box width
inner_gap     = 0.012      # gap between composition boxes within a group
group_centres = np.arange(len(EL_ORDER)) * (
    n_comp * (box_width + inner_gap) + group_spacing)

for ge, el in enumerate(EL_ORDER):
    for ic, row in comp_order.iterrows():
        x_off = (ic - (n_comp - 1) / 2) * (box_width + inner_gap)
        x_pos = group_centres[ge] + x_off
        sub = b[(b['element'] == el) & (b['comp_label'] == row['comp_label'])]
        vals = sub['mu_PAW'].abs().values
        if len(vals) == 0:
            continue
        ax.boxplot(vals, positions=[x_pos], widths=box_width,
                   patch_artist=True, showfliers=True,
                   flierprops=dict(marker='.', markersize=2,
                                   markerfacecolor=comp_colours[row['comp_label']],
                                   markeredgecolor='none', alpha=0.7),
                   medianprops=dict(color='black', linewidth=0.7),
                   whiskerprops=dict(linewidth=0.6, color='#444444'),
                   capprops=dict(linewidth=0.6, color='#444444'),
                   boxprops=dict(linewidth=0.5,
                                 edgecolor='#222222',
                                 facecolor=comp_colours[row['comp_label']],
                                 alpha=0.85))

# Mn assignment bands + threshold lines
ax.axhspan(MN43_BAND, MN32_BAND, color='#f0d4b8', alpha=0.40, zorder=0)
ax.axhspan(MN32_BAND, 4.7,       color='#e8b8b8', alpha=0.35, zorder=0)
ax.axhline(MN43_BAND, color='#888888', lw=0.6, ls=':', zorder=0.5)
ax.axhline(MN32_BAND, color='#888888', lw=0.6, ls=':', zorder=0.5)
ax.axhline(CUTOFF_CE, color='#888888', lw=0.6, ls=':', zorder=0.5)

# Right-edge assignment labels for Mn bands
xmax_a = group_centres[-1] + n_comp * (box_width + inner_gap) / 2 + 0.10
ax.text(xmax_a, (MN43_BAND + MN32_BAND) / 2, ' Mn$^{3+}$',
        va='center', ha='left', fontsize=7, color='#555555')
ax.text(xmax_a, (MN32_BAND + 4.5) / 2,       ' Mn$^{2+}$',
        va='center', ha='left', fontsize=7, color='#555555')
ax.text(xmax_a, MN43_BAND - 0.20,            ' Mn$^{4+}$',
        va='center', ha='left', fontsize=7, color='#555555')
# Ce4+ cutoff label placed inside the panel in the empty Ti-Mn gap, just above
# the cutoff line. The bbox masks the dotted line behind the glyphs.
ce_label_x = (group_centres[2] + group_centres[3]) / 2
ax.text(ce_label_x, CUTOFF_CE + 0.10, r'Ce$^{4+}$ cutoff',
        va='bottom', ha='center', fontsize=7, color='#555555',
        bbox=dict(facecolor='white', alpha=0.8,
                  edgecolor='none', pad=1.2))

ax.set_xticks(group_centres)
ax.set_xticklabels(EL_ORDER)
ax.set_xlim(group_centres[0] - n_comp * (box_width + inner_gap) / 2 - 0.10,
            xmax_a + 0.65)
ax.set_ylim(-0.20, 4.7)
ax.set_ylabel(r'$|\mu|$ ($\mu_\mathrm{B}$)')
ax.set_xlabel('Element')

# Composition legend below the panel, 2 columns x 3 rows. Placed so the
# white space between the legend top and the bottom of the x-axis label
# equals the space between the x-axis label and the x-tick labels
# (measured from the drawn canvas; same convention as figure3.py).
handles_comp = [Patch(facecolor=comp_colours[r['comp_label']],
                      edgecolor='#222222', linewidth=0.5,
                      label=formula_to_label(r['formula']))
                for _, r in comp_order.iterrows()]
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
xlab_bb = ax.xaxis.label.get_window_extent(renderer)
ticks_y0 = min(t.get_window_extent(renderer).y0
               for t in ax.get_xticklabels())
gap_px = ticks_y0 - xlab_bb.y1
leg_top_fig = (xlab_bb.y0 - gap_px) / fig.bbox.height
fig.legend(handles=handles_comp, loc='upper center',
           ncol=2, frameon=False, fontsize=6.5,
           handletextpad=0.4, columnspacing=1.0,
           handlelength=1.3, bbox_to_anchor=(0.5, leg_top_fig))

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.close(fig)

# ---------- Console summary: numbers cited in prose + the ESI Ce-count table ----------
print('=== Validation paragraph numbers ===')
mad_bulk = np.mean(np.abs(b['mu_Bader'].abs() - b['mu_PAW'].abs()))
print(f'  Bulk |mu| PAW-vs-Bader: N={len(b)}, MAD={mad_bulk:.4f} uB')

print(f'\n=== ESI table: Ce 3+/4+ counts (cutoff = {CUTOFF_CE:.1f} uB) ===')
print(f"  {'composition':<35s}  x_Ce   x_Mn   N_Ce  N_Ce3+  N_Ce4+")
for _, row in comp_order.iterrows():
    sub = b[(b['comp_label'] == row['comp_label']) & (b['element'] == 'Ce')]
    n_ce = len(sub)
    n3   = int((sub['mu_PAW'].abs() > CUTOFF_CE).sum())
    n4   = n_ce - n3
    print(f"  {row['formula']:<35s}  {row['x_Ce']:.3f}  {row['x_Mn']:.3f}  "
          f"{n_ce:4d}   {n3:4d}    {n4:4d}")

print(f"Wrote {OUT_PNG}")
