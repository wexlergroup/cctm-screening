"""Render the full unit cell with BO6 coordination polyhedra."""

from ovito.io import import_file
from ovito.vis import Viewport, OSPRayRenderer
from render_utils import (
    append_central_cell_trim,
    append_element_colors,
    append_global_bonds,
    append_polyhedra,
    append_replicate,
    element_color,
    element_in_central_cell_expr,
    get_element_type_ids,
    make_color_legend_overlay,
    make_coordinate_tripod_overlay,
)
from style_params import COLOR_DICT

# Elements that get an octahedron mesh drawn for every instance inside the
# central cell.
POLYHEDRA_ELEMENTS = ("Ti", "Mn")
POLYHEDRA_TRANSPARENCY = 0.25

# Color legend painted onto the render: A-site (blue, Ca), B-site (gray, Ti),
# O (red). Drawn left-to-right in this order.
COLOR_LEGEND = (
    ("A", COLOR_DICT["Ca"]),
    ("B", COLOR_DICT["Ti"]),
    ("O", COLOR_DICT["O"]),
)


def render_structure(input_path, output_path, size=(1000, 1000)):

    # ==== Set up pipeline ==== #

    pipeline = import_file(input_path)
    present_elements = set(get_element_type_ids(pipeline.compute()))

    append_global_bonds(pipeline)
    append_replicate(pipeline)

    # Build polyhedra BEFORE trimming so boundary Ti/Mn still see their full
    # oxygen shell. Centers are restricted to the central cell so we don't
    # spawn polyhedra in the 26 surrounding replicated cells.
    for element in POLYHEDRA_ELEMENTS:
        if element not in present_elements:
            continue
        append_polyhedra(
            pipeline,
            element_color(element),
            selection_expression=element_in_central_cell_expr(element),
            transparency=POLYHEDRA_TRANSPARENCY,
        )

    # Trim to {central-cell cations} ∪ {their bonded O ligands}. Polyhedra
    # SurfaceMesh from above is unaffected by particle deletion.
    append_central_cell_trim(pipeline)

    append_element_colors(pipeline, get_element_type_ids(pipeline.compute()))

    # ======= Set up visual elements ======= #

    data = pipeline.compute()
    data.particles.vis.scaling = 0.5
    data.cell.vis.enabled = False

    # Second pipeline for the unit cell wireframe only
    cell_pipeline = import_file(input_path)
    cell_pipeline.compute().particles.vis.enabled = False

    pipeline.add_to_scene()
    cell_pipeline.add_to_scene()

    # ========== Set up rendering ========== #

    # Type.Top is the built-in straight-down-Z orthographic preset — parallel
    # octahedral edges stay parallel, so the tiling pattern reads clearly.
    # zoom_all auto-frames the scene.
    vp = Viewport(type=Viewport.Type.Top)
    vp.zoom_all(size=size)
    vp.overlays.append(make_coordinate_tripod_overlay())
    vp.overlays.append(make_color_legend_overlay(COLOR_LEGEND))

    renderer = OSPRayRenderer(
        samples_per_pixel=32,
        ambient_brightness=1.2,
        material_type="principled",
    )

    vp.render_image(filename=output_path, size=size, renderer=renderer)


def main():
    # Renders the panel that figure1.py drops in as Fig. 1a. Alternative
    # inputs (the six CCTM supercells) would render the same way; only the
    # idealized CaTiO3 cell is used for the published panel.
    # data/cctm2112_sqs.cif is the committed CCTM2112 example.
    render_structure(
        "data/catio3_unit_cell.cif",
        "panel_1a_bulk_structure.png",
        size=(2000, 2000),   # ~5.4 cm panel at 600 dpi needs >= 1280 px
    )


if __name__ == "__main__":
    main()
