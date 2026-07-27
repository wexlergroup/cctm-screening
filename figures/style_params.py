"""Element colors for the OVITO structure-render scripts (render_*.py).

Single source for the per-element colors used in the Fig. 1 panels; the
fgr:1 caption in main.tex names these colors in prose (Ca blue, Ce green,
Ti gray, Mn purple, O red), so a change here must be mirrored there.

This module deliberately carries no matplotlib styling: per the repo
convention, each figure script holds its own inline rcParams block.
"""

COLOR_DICT: dict[str, str] = {
    "Ca": "#5a96bd",
    "Ce": "#5ac449",
    "Ti": "#878681",
    "Mn": "#a8089e",
    "O": "#fe0300",
}
