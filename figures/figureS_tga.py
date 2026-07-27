"""ESI TGA figure: Protocol B redox cycling for the three synthesized CCTM
compositions, three vertical panels (single column, esi.tex label fgr:tga).

Each panel shows a complete 18-hour TGA run: sample temperature (red, left
y-axis) and Δδ derived from buoyancy-corrected mass loss (black, right
y-axis). Δδ references the most-oxidized state in the run to zero.

Protocol A: 1350 °C (0.5 h) -> 1300 °C (1 h) reduction in Ar*;
            900 °C (0.5 h) re-oxidation in 20% O2/Ar.
Protocol B: 1350 °C (1 h)   -> 1300 °C (0.5 h) reduction in Ar*;
            900 °C (0.5 h) re-oxidation in 20% O2/Ar.

Reads:  data/tga_CCTM2112_protocol_B.csv, data/tga_CCTM3122_protocol_B.csv,
        data/tga_CCTM1771014_protocol_B.csv
Writes: plots/figureS_tga.png (600 dpi)

Run from inside ``figures/``:

    python figureS_tga.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import fig_style

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
OUT_PNG = PLOT_DIR / 'figureS_tga.png'

# Map: (filename, composition full label, X_Mn, protocol letter)
RUNS = [
    ('tga_CCTM2112_protocol_B.csv',
     r'Ca$_{0.67}$Ce$_{0.33}$Ti$_{0.33}$Mn$_{0.67}$O$_3$', 0.67, 'B'),
    ('tga_CCTM3122_protocol_B.csv',
     r'Ca$_{0.75}$Ce$_{0.25}$Ti$_{0.50}$Mn$_{0.50}$O$_3$', 0.50, 'B'),
    ('tga_CCTM1771014_protocol_B.csv',
     r'Ca$_{0.71}$Ce$_{0.29}$Ti$_{0.42}$Mn$_{0.58}$O$_3$', 0.58, 'B'),
]

# Composition order for stacked panels (low to high X_Mn).
COMPS_ORDERED = sorted({(r[1], r[2]) for r in RUNS}, key=lambda x: x[1])

T_COLOR = '#c0392b'        # temperature: red
DD_COLOR = 'black'         # Delta-delta: black


def load_run(fn):
    df = pd.read_csv(DATA_DIR / fn)
    t = df["time_h"].values       # h
    T = df["temperature_C"].values     # deg C
    lowest_delta = df.iloc[:, -1].min()
    dd = df.iloc[:, -1].values - lowest_delta
    return t, T, dd


# Shared y limits across all panels so cross-panel comparison is honest.
# The headroom above the highest curve (1350 °C plateau on the left axis,
# the CCTM2112 first-cycle anomaly on the right) holds the composition label.
T_LO, T_HI = 0, 1620
DD_LO, DD_HI = -0.005, 0.155


def style_panel(ax_T, ax_dd):
    ax_T.set_ylim(T_LO, T_HI)
    ax_T.set_yticks([0, 500, 1000, 1350])
    ax_T.tick_params(axis='y', colors=T_COLOR)
    ax_T.spines['left'].set_color(T_COLOR)

    ax_dd.set_ylim(DD_LO, DD_HI)
    ax_dd.set_yticks([0.00, 0.025, 0.05, 0.075, 0.10, 0.125])
    ax_dd.tick_params(axis='y', colors=DD_COLOR)
    ax_dd.spines['right'].set_color(DD_COLOR)
    ax_dd.spines['left'].set_color(T_COLOR)
    ax_dd.spines['top'].set_visible(False)


def render():
    W = fig_style.cm2in(fig_style.EES_SINGLE_CM)
    fig, axes = plt.subplots(3, 1, figsize=(W, W * 1.50),
                             sharex=True, dpi=600)

    dd_axes = []
    for ax_T, (comp_label, xMn) in zip(axes, COMPS_ORDERED):
        fn = [r[0] for r in RUNS if r[1] == comp_label][0]
        t, T, dd = load_run(fn)
        ax_dd = ax_T.twinx()
        dd_axes.append(ax_dd)
        ax_T.plot(t, T, color=T_COLOR, lw=0.6)
        ax_dd.plot(t, dd, color=DD_COLOR, lw=0.7)
        style_panel(ax_T, ax_dd)

        # Composition label centered in the headroom between the highest
        # curve in the panel and the top border (equal white space above
        # and below the label).
        curve_top = max((np.nanmax(T) - T_LO) / (T_HI - T_LO),
                        (np.nanmax(dd) - DD_LO) / (DD_HI - DD_LO))
        ax_T.text(0.02, (curve_top + 1.0) / 2.0, comp_label,
                  transform=ax_T.transAxes, ha='left', va='center',
                  fontsize=7)

    axes[-1].set_xlabel('Time (h)')
    axes[-1].set_xlim(0, 19)
    axes[-1].set_xticks([0, 4, 8, 12, 16])

    lab_T = fig.text(0.0, 0.5, 'Temperature (°C)', va='center',
                     rotation='vertical', color=T_COLOR, fontsize=8)
    lab_dd = fig.text(1.0, 0.5, r'$\Delta\delta$', va='center',
                      rotation='vertical', color=DD_COLOR, fontsize=8)

    fig.tight_layout(rect=[0.04, 0, 0.96, 1.0])

    # Pull the shared y-axis labels in to a fixed 4 pt pad from the
    # outermost tick labels (measured from the drawn canvas).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad_px = 4 / 72 * fig.dpi
    ticks_x0 = min(t.get_window_extent(renderer).x0
                   for ax in axes for t in ax.get_yticklabels())
    bb = lab_T.get_window_extent(renderer)
    lab_T.set_x(lab_T.get_position()[0]
                + (ticks_x0 - pad_px - bb.x1) / fig.bbox.width)
    ticks_x1 = max(t.get_window_extent(renderer).x1
                   for ax in dd_axes for t in ax.get_yticklabels())
    bb = lab_dd.get_window_extent(renderer)
    lab_dd.set_x(lab_dd.get_position()[0]
                 + (ticks_x1 + pad_px - bb.x0) / fig.bbox.width)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
    plt.close(fig)


render()
print(f"Wrote {OUT_PNG}")
