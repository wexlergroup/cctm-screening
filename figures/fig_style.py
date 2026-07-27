"""Shared EES figure style for all figure scripts in this directory.

Importing this module applies the canonical rcParams (decided in the
2026-06-04 consistency pass): Arial-first sans stack, 8 pt body / 7 pt tick
and legend text, 0.7 pt axes and tick lines, inward 3.0 pt ticks, and the
custom mathtext setup that keeps math variables italic while `\\mathrm{}`
subscripts stay upright. Without ``mathtext.fontset = 'custom'``, matplotlib
silently renders every ``$...$`` in DejaVu math glyphs even when the body
font is Arial. Regenerate figures only on a machine where Arial resolves
(check stderr for ``findfont`` warnings).

Scripts keep any figure-specific rcParams as explicit overrides after the
import, so the effective style of each committed PNG is unchanged.
"""
import matplotlib.pyplot as plt

# EES column widths (cm) and the cm -> inch conversion used for figsize.
EES_SINGLE_CM = 8.5
EES_DOUBLE_CM = 17.1
CM_PER_INCH = 2.54


def cm2in(cm):
    """Convert a length in cm to inches for matplotlib figsize."""
    return cm / CM_PER_INCH


# Model colors (shared across the model-comparison figures).
COLOR_DGNN = '#2b6cb0'
COLOR_CFM = '#c0392b'
COLOR_DGNN_FT = '#e69500'

RC_PARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'mathtext.default': 'it',
    'mathtext.fontset': 'custom',
}

plt.rcParams.update(RC_PARAMS)
