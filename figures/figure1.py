"""Figure 1: perovskite bulk structure, local O environment, and sampled composition space.

Three-panel double-column figure: (a) OVITO bulk-structure rendering,
(b) OVITO local oxygen nearest-neighbor rendering, (c) the six bulk DFT
compositions in (X_Ce, X_Mn) space, on the two series that intersect at
CCTM2112 (X_Mn = 2 X_Ce and X_Ce = 1/3). If the OVITO PNGs are missing,
panels (a) and (b) are drawn as dashed placeholder boxes.

Reads:
    data/bulk_compositions.csv
    panel_1a_bulk_structure.png   (OVITO render, from render_unit_cell.py)
    panel_1b_local_environment.png (OVITO render, from render_o_site.py)
Writes:
    plots/figure1.png
Run from inside figures/: python3 figure1.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

import fig_style


# ---------- Configuration ----------
PANEL_A_PNG = Path('panel_1a_bulk_structure.png')      # OVITO bulk render
PANEL_B_PNG = Path('panel_1b_local_environment.png')   # OVITO local zoom

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')

DATA_CSV = DATA_DIR / 'bulk_compositions.csv'
OUT_PNG = PLOT_DIR / 'figure1.png'

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Panel renderers ----------
def autocrop(img: np.ndarray, pad: int = 12) -> np.ndarray:
    """Trim uniform white/transparent borders from a render, keeping `pad` px.

    The OVITO renders frame a small object in a square viewport, leaving wide
    blank margins; cropping lets each panel fill its slot."""
    content = np.any(img[..., :3] < 0.98, axis=-1)
    if img.shape[-1] == 4:
        content &= img[..., 3] > 0.01
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0 = max(rows[0] - pad, 0)
    r1 = min(rows[-1] + pad, img.shape[0] - 1)
    c0 = max(cols[0] - pad, 0)
    c1 = min(cols[-1] + pad, img.shape[1] - 1)
    return img[r0:r1 + 1, c0:c1 + 1]


def render_image_or_placeholder(ax, image_path: Path, slot_label: str,
                                 slot_desc: str) -> None:
    """Insert an external PNG into `ax`, or draw a dashed placeholder if absent."""
    if image_path.is_file():
        img = autocrop(mpimg.imread(image_path))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.text(0.5, 0.5,
                f'{slot_label}: {slot_desc}\n\nplace OVITO PNG at\n{image_path.name}',
                ha='center', va='center', fontsize=7, color='0.45',
                transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_edgecolor('0.65')
            spine.set_linestyle((0, (4, 4)))
            spine.set_linewidth(0.8)
        ax.set_xticks([])
        ax.set_yticks([])


def render_composition_space(ax) -> None:
    """Panel 1c: scatter of the six bulk DFT compositions in (X_Ce, X_Mn) space."""
    keep = ['M', 'N', 'composition', 'x_Ce', 'x_Mn', 'ehull_in_meV', 'avgEv_in_eV']
    df = pd.read_csv(DATA_CSV, usecols=keep)

    atol = 1e-3
    on_A = (df['x_Mn'] - 2 * df['x_Ce']).abs() < atol
    on_B = (df['x_Ce'] - 1 / 3).abs() < atol
    df['group'] = np.where(on_A & on_B, 'intersection',
                    np.where(on_A, 'series_A',
                    np.where(on_B, 'series_B', 'other')))

    c_A, c_B, c_X = '#1f6cb5', '#c0392b', '#e8b923'

    # Guide lines
    xx = np.linspace(0, 0.5, 200)
    ax.plot(xx, 2 * xx, color=c_A, lw=0.9, alpha=0.55, zorder=1,
            label=r'$X_\mathrm{Mn} = 2\,X_\mathrm{Ce}$')
    ax.axvline(1 / 3, color=c_B, lw=0.9, alpha=0.55, zorder=1,
               label=r'$X_\mathrm{Ce} = 1/3$')

    # Points
    styles = {
        'series_A':    dict(c=c_A, marker='o', s=45),
        'series_B':    dict(c=c_B, marker='s', s=45),
        'intersection': dict(c=c_X, marker='D', s=80),
    }
    for grp, kw in styles.items():
        sub = df[df['group'] == grp]
        ax.scatter(sub['x_Ce'], sub['x_Mn'],
                   edgecolors='black', linewidths=0.6, zorder=3, **kw)

    # Manual legend entry for intersection (a marker, not a line)
    handles, labels = ax.get_legend_handles_labels()
    int_label = r'$(X_\mathrm{Ce},X_\mathrm{Mn})=(1/3,\,2/3)$'
    handles.append(Line2D([0], [0], marker='D', color='none',
                          markerfacecolor=c_X, markeredgecolor='black',
                          markeredgewidth=0.6, markersize=7,
                          label=int_label))
    labels.append(int_label)

    ax.set_xlabel(r'$X_\mathrm{Ce}$')
    ax.set_ylabel(r'$X_\mathrm{Mn}$')
    ax.set_xlim(0.215, 0.405)
    ax.set_ylim(0.45, 0.79)
    ax.set_xticks([0.25, 0.30, 0.35, 0.40])
    ax.set_yticks([0.50, 0.60, 0.70])
    ax.set_box_aspect(1)   # square axes box, independent of the data limits

    leg = ax.legend(handles=handles, labels=labels,
                    loc='lower right', frameon=True, framealpha=0.95,
                    edgecolor='0.6', borderpad=0.4, handletextpad=0.5,
                    handlelength=1.6)
    leg.get_frame().set_linewidth(0.5)


def panel_label(ax, text: str) -> None:
    """Bold (a)/(b)/(c) label in the upper-left corner of a panel."""
    ax.text(-0.02, 1.02, text, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='left')


# ---------- Assemble the composite ----------
W = fig_style.cm2in(fig_style.EES_DOUBLE_CM)   # inches
H = W / 3.1                      # roughly square panels
fig, axes = plt.subplots(1, 3, figsize=(W, H), dpi=600,
                         gridspec_kw={'width_ratios': [1.0, 0.5, 1.15],
                                      'wspace': 0.18})

render_image_or_placeholder(axes[0], PANEL_A_PNG,
                            slot_label='1a', slot_desc='bulk structure')
panel_label(axes[0], '(a)')

render_image_or_placeholder(axes[1], PANEL_B_PNG,
                            slot_label='1b', slot_desc='local O environment')
panel_label(axes[1], '(b)')

render_composition_space(axes[2])
panel_label(axes[2], '(c)')

# ---------- Enlarge panel (a): top-left anchored, fixed aspect ----------
# Pull (a)'s bottom-right corner down/right until the buffer between its
# bottom spine and the figure bottom (tight bbox) equals the buffer between
# its left spine and the figure left edge. Iterated because the tight bbox
# itself moves once (a)'s bottom becomes the lowest content.
renderer = fig.canvas.get_renderer()
for _ in range(6):
    fig.canvas.draw()
    tight = fig.get_tightbbox(renderer)            # figure inches
    pos = axes[0].get_position(original=False)     # figure fractions
    x0_in, y0_in = pos.x0 * W, pos.y0 * H
    w_in, h_in = pos.width * W, pos.height * H
    left_buf = x0_in - tight.x0
    target_bottom = tight.y0 + left_buf            # bottom buffer == left buffer
    if abs(y0_in - target_bottom) < 1e-3:
        break
    s = (y0_in + h_in - target_bottom) / h_in      # scale, top edge fixed
    axes[0].set_position([pos.x0, (y0_in + h_in - s * h_in) / H,
                          s * w_in / W, s * h_in / H])

# ---------- Recenter panel (b) between its visual neighbors ----------
# An imshow panel centers in its own gridspec slot, which ignores (a)'s
# aspect-narrowed drawn box and (c)'s y-axis label protruding into the gap.
# Shift (b) to the midpoint between (a)'s right spine and the ylabel's left
# edge, measured after a draw so aspect boxes and text extents are realized.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
a_right = axes[0].get_position(original=False).x1
b_box = axes[1].get_position(original=False)
ylab_extent = axes[2].yaxis.label.get_window_extent(renderer)
ylab_left = fig.transFigure.inverted().transform((ylab_extent.x0, 0))[0]
shift = 0.5 * (a_right + ylab_left) - 0.5 * (b_box.x0 + b_box.x1)
axes[1].set_position([b_box.x0 + shift, b_box.y0, b_box.width, b_box.height])

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
print(f"Wrote {OUT_PNG}")
