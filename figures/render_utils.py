"""Shared OVITO pipeline helpers for the CCTM structure-render scripts.

Thin wrapper around `ovito.modifiers` that centralises the patterns reused
across render-*.py scripts: a standard 3x3x3 replication, central-cell
trimming via bond-based selection expansion, consistent polyhedra styling,
and per-element coloring keyed off `style_params.COLOR_DICT`.
"""

import os
from ovito.modifiers import (
    AssignColorModifier,
    ClearSelectionModifier,
    CoordinationPolyhedraModifier,
    CreateBondsModifier,
    DeleteSelectedModifier,
    ExpandSelectionModifier,
    ExpressionSelectionModifier,
    InvertSelectionModifier,
    ReplicateModifier,
    SelectTypeModifier,
)
from ovito.vis import CoordinateTripodOverlay
from style_params import COLOR_DICT


# Shared pipeline defaults
N_REP = 3
CENTRAL_CELL_MARGIN = 0.05
BOND_CUTOFF = 2.2
BOND_WIDTH = 0.2


# ---------- Color helpers ---------- #

def hex_to_rgb_tuple(hex_color: str) -> tuple:
    """Convert '#RRGGBB' (or 'RRGGBB') to an RGB tuple of floats in [0, 1]."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Unexpected hex color length for '{hex_color}'.")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def element_color(element: str) -> list:
    """RGB list for `element`, sourced from style_params.COLOR_DICT."""
    return list(hex_to_rgb_tuple(COLOR_DICT[element]))


def get_element_type_ids(data) -> dict:
    """{element_name: type_id} mapping for particle types in `data`."""
    return {t.name: t.id for t in data.particles.particle_types.types}


# ---------- Expression fragments ---------- #

def _central_cell_inside_expr(n_rep: int, margin: float) -> str:
    lo = round((1 - margin) / n_rep, 4)
    hi = round((n_rep - 1 + margin) / n_rep, 4)
    return " && ".join(
        f"(ReducedPosition.{ax} >= {lo} && ReducedPosition.{ax} <= {hi})"
        for ax in "XYZ"
    )


def element_in_central_cell_expr(
    element: str,
    n_rep: int = N_REP,
    margin: float = CENTRAL_CELL_MARGIN,
) -> str:
    """Expression selecting only `element` atoms inside the central cell
    of an n_rep^3 supercell."""
    return f'ParticleType == "{element}" && ({_central_cell_inside_expr(n_rep, margin)})'


def cations_in_central_cell_expr(
    n_rep: int = N_REP,
    margin: float = CENTRAL_CELL_MARGIN,
) -> str:
    """Expression selecting non-oxygen atoms inside the central cell."""
    return f'ParticleType != "O" && ({_central_cell_inside_expr(n_rep, margin)})'


# ---------- Pipeline builders ---------- #

def append_global_bonds(
    pipeline,
    cutoff: float = BOND_CUTOFF,
    width: float = BOND_WIDTH,
) -> None:
    """Append a CreateBondsModifier with a single cutoff over all type pairs."""
    mod = CreateBondsModifier()
    mod.vis.width = width
    mod.cutoff = cutoff
    pipeline.modifiers.append(mod)


def append_replicate(pipeline, n_rep: int = N_REP) -> None:
    """Replicate n_rep^3 with adjust_box=True (OVITO default)."""
    pipeline.modifiers.append(ReplicateModifier(num_x=n_rep, num_y=n_rep, num_z=n_rep))


def append_central_cell_trim(
    pipeline,
    n_rep: int = N_REP,
    margin: float = CENTRAL_CELL_MARGIN,
) -> None:
    """Trim to {central-cell cations} ∪ {their bonded O ligands}.

    Seeds the selection with cations inside the central cell, grows by one
    bond to pick up their O ligands (including ones that sit past the cell
    wall), then deletes the complement. Relies on bonds having been created
    earlier in the pipeline.
    """
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression=cations_in_central_cell_expr(n_rep, margin)))
    pipeline.modifiers.append(ExpandSelectionModifier(
        mode=ExpandSelectionModifier.ExpansionMode.Bonded,
        iterations=1,
    ))
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())


def append_polyhedra(
    pipeline,
    color,
    *,
    selection_expression: str = None,
    selection_seeder=None,
    transparency: float = 0.25,
) -> None:
    """Append a CoordinationPolyhedraModifier, then ClearSelection.

    At most one of `selection_expression` (str) or `selection_seeder` (a
    modifier or Python modifier fn) may be passed to seed the centers; if
    neither is given the current pipeline selection is used as-is.
    """
    if selection_expression is not None and selection_seeder is not None:
        raise ValueError("Pass only one of selection_expression or selection_seeder.")
    if selection_expression is not None:
        pipeline.modifiers.append(ExpressionSelectionModifier(expression=selection_expression))
    elif selection_seeder is not None:
        pipeline.modifiers.append(selection_seeder)

    mod = CoordinationPolyhedraModifier()
    mod.vis.surface_color = list(color)
    mod.vis.surface_transparency = transparency
    mod.vis.highlight_edges = True
    pipeline.modifiers.append(mod)
    pipeline.modifiers.append(ClearSelectionModifier())


def append_element_colors(pipeline, element_type_ids: dict) -> None:
    """Color each element in `element_type_ids` according to COLOR_DICT."""
    for element, type_id in element_type_ids.items():
        pipeline.modifiers.append(SelectTypeModifier(types={type_id}))
        pipeline.modifiers.append(AssignColorModifier(color=element_color(element)))
        pipeline.modifiers.append(ClearSelectionModifier())


# ---------- Visuals ---------- #

def make_coordinate_tripod_overlay() -> CoordinateTripodOverlay:
    """Standard XYZ-tripod overlay used across the render scripts."""
    return CoordinateTripodOverlay(
        axis1_label="<i>a</i>",
        axis2_label="<i>b</i>",
        axis3_label="<i>c</i>",
    )


def make_color_legend_overlay(
    entries,
    *,
    y_frac: float = 0.08,
    dot_radius_frac: float = 0.022,
    entry_stride_frac: float = 0.11,
    font_size_frac: float = 0.028,
):
    """Return a PythonViewportOverlay that paints a horizontal color legend
    onto the rendered image: one filled circle per entry, with the entry's
    label drawn to its right.

    Parameters
    ----------
    entries : sequence of (label, color)
        Drawn left-to-right. `color` may be any value accepted by
        ``QtGui.QColor`` — a '#RRGGBB' hex string is the typical choice.
    y_frac : float
        Vertical position of the dot centers, as a fraction of image height
        (0 = top, 1 = bottom). Default puts the legend near the top.
    dot_radius_frac, entry_stride_frac, font_size_frac : float
        Sizes relative to ``min(width, height)`` of the viewport, so the
        legend scales with image resolution.
    """
    # Imported inside the function so `render_utils` stays importable even
    # if Qt bindings aren't available in a given environment.
    from ovito.qt_compat import QtCore, QtGui
    from ovito.vis import PythonViewportOverlay

    def render(args):
        painter = args.painter
        width, height = args.size
        base = min(width, height)
        radius = base * dot_radius_frac
        stride = base * entry_stride_frac
        y = height * y_frac

        font = painter.font()
        font.setPointSizeF(base * font_size_frac)
        painter.setFont(font)

        row_width = (len(entries) - 1) * stride
        start_x = (width - row_width) / 2

        painter.setPen(QtGui.QPen(QtGui.QColor("black"), 1.0))
        label_align = int(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        for i, (label, color) in enumerate(entries):
            cx = start_x + i * stride
            painter.setBrush(QtGui.QBrush(QtGui.QColor(color)))
            painter.drawEllipse(QtCore.QPointF(cx, y), radius, radius)
            # Label rect to the right of the dot, vertically centered on it
            label_rect = QtCore.QRectF(
                cx + radius * 1.5, y - radius, base * 0.1, radius * 2
            )
            painter.drawText(label_rect, label_align, label)

    overlay = PythonViewportOverlay()
    overlay.function = render
    return overlay


# ---------- Driver ---------- #

def render_first_missing(input_paths, render_fn, output_suffix: str) -> None:
    """Iterate `input_paths`; for the first one whose output doesn't exist,
    call `render_fn(input, output)` and stop. Matches the pattern used by
    the per-figure render scripts."""
    for input_path in input_paths:
        output_path = input_path.replace(".cif", output_suffix)
        if os.path.exists(output_path):
            print(f"Skipping {input_path} (already rendered as {output_path})")
            continue
        render_fn(input_path, output_path)
        break
