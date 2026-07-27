"""Render the local environment of a named oxygen atom (TARGET_O_FRAC):
the O itself, its first nearest-neighbor cations, the O-cation bonds,
and the B-site coordination octahedra that share this O as a corner."""

import math
import numpy as np
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, DeleteSelectedModifier
from ovito.data import DataCollection
from ovito.vis import Viewport, OSPRayRenderer
from render_utils import (
    BOND_WIDTH,
    N_REP,
    append_central_cell_trim,
    append_element_colors,
    append_global_bonds,
    append_polyhedra,
    append_replicate,
    element_color,
    get_element_type_ids,
    make_coordinate_tripod_overlay,
)

# Elements that get a coordination-octahedron mesh drawn for each of their
# first-neighbor B-sites of the target O. Ti is omitted intentionally —
# in a CaTiO3-only cell the Ti octahedron would sit on top of the figure
# and obscure the bonds and vertex oxygens we're highlighting. Add "Ti"
# back here for a different render.
POLYHEDRA_ELEMENTS = ("Mn",)
POLYHEDRA_TRANSPARENCY = 0.5

# Pairwise cation-O cutoffs (Å) on top of the global 2.2 Å pass, which only
# captures short B-O bonds. A-site M-O bonds run ~2.4-2.7 Å.
A_SITE_PAIR_CUTOFFS = {"Ca": 3.1, "Ce": 3.1}

# Fractional (original-cell) position of the target O atom — "30 (O3_3_1)"
# in OVITO's GUI, i.e. _atom_site_label == "O3_3_1" in the CaTiO3 B-face
# centred CIF. After 3x3x3 replication with adjust_box=True the central
# copy of this atom lands at supercell reduced position (1 + frac) / n_rep.
TARGET_O_FRAC = (0.492223, 0.500000, 0.721286)
# Max distance (Å) allowed between the target position and the matched atom;
# catches silent mismatches if the CIF changes or the replication shifts.
TARGET_O_MATCH_TOL = 0.2


def _find_target_o_index(data: DataCollection) -> int:
    """Index of the target O atom — the copy of TARGET_O_FRAC that sits in
    the central cell. Matched by Cartesian distance so the lookup survives
    replication, trimming, and any reordering upstream in the pipeline."""
    positions = np.asarray(data.particles.positions)
    type_array = np.asarray(data.particles.particle_types)
    o_type = next(
        t.id for t in data.particles.particle_types.types if t.name == "O"
    )
    o_mask = type_array == o_type
    o_global = np.where(o_mask)[0]
    if len(o_global) == 0:
        raise RuntimeError("No oxygen atoms found in the structure.")

    cell = np.asarray(data.cell)  # 3x4: [a | b | c | origin]
    target_reduced_super = (1.0 + np.asarray(TARGET_O_FRAC)) / N_REP
    target_cart = cell[:, :3] @ target_reduced_super + cell[:, 3]

    dists = np.linalg.norm(positions[o_mask] - target_cart, axis=1)
    best = int(np.argmin(dists))
    if dists[best] > TARGET_O_MATCH_TOL:
        raise RuntimeError(
            f"No O atom within {TARGET_O_MATCH_TOL} Å of "
            f"TARGET_O_FRAC={TARGET_O_FRAC}; closest O is "
            f"{dists[best]:.3f} Å away. Check the CIF, or widen "
            "TARGET_O_MATCH_TOL."
        )
    return int(o_global[best])


def _bonded_neighbors(data: DataCollection, particle_idx: int) -> set:
    bonds = data.particles.bonds
    if bonds is None or bonds.count == 0:
        return set()
    topo = np.asarray(bonds.topology)
    attached = (topo[:, 0] == particle_idx) | (topo[:, 1] == particle_idx)
    partners = topo[attached].ravel()
    return {int(p) for p in partners if int(p) != particle_idx}


def _make_b_neighbor_selector(elements):
    """Select the target O's bonded neighbors whose type is in `elements`.
    Used as the CoordinationPolyhedra seed so only the octahedra sharing
    the target O are generated."""

    def modifier(_frame: int, data: DataCollection):
        target_o = _find_target_o_index(data)
        type_ids = {
            t.id for t in data.particles.particle_types.types if t.name in elements
        }
        type_array = np.asarray(data.particles.particle_types)
        sel = np.zeros(data.particles.count, dtype=np.int8)
        for n in _bonded_neighbors(data, target_o):
            if int(type_array[n]) in type_ids:
                sel[n] = 1
        data.particles_.create_property("Selection", data=sel)

    return modifier


def _make_environment_selector():
    """Select atoms NOT in the target-O environment (target O + its
    first-neighbor cations). A downstream DeleteSelected strips the rest
    so only the isolated site remains."""

    def modifier(_frame: int, data: DataCollection):
        target_o = _find_target_o_index(data)
        keep = {target_o} | _bonded_neighbors(data, target_o)
        sel = np.ones(data.particles.count, dtype=np.int8)
        sel[list(keep)] = 0
        data.particles_.create_property("Selection", data=sel)

    return modifier


def _append_a_site_pairwise_bonds(pipeline, present_elements):
    """Pairwise A-site M-O bonds (Ca-O, Ce-O ~ 2.4-2.7 Å). Call after the
    central-cell trim so we only grow bonds between atoms we're keeping."""
    pair_mod = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    pair_mod.vis.width = BOND_WIDTH
    for cation, cutoff in A_SITE_PAIR_CUTOFFS.items():
        if cation in present_elements:
            pair_mod.set_pairwise_cutoff("O", cation, cutoff)
    pipeline.modifiers.append(pair_mod)


def render_o_site(input_path, output_path, size=(1000, 1000)):

    # ==== Set up pipeline ==== #

    pipeline = import_file(input_path)
    element_type_ids = get_element_type_ids(pipeline.compute())
    present_elements = set(element_type_ids)

    append_global_bonds(pipeline)
    append_replicate(pipeline)
    append_central_cell_trim(pipeline)
    _append_a_site_pairwise_bonds(pipeline, present_elements)

    # Build an octahedron mesh for each B-site neighbor of the target O.
    # Must happen BEFORE the isolation step — once we delete the surrounding
    # O ligands, CoordinationPolyhedra would produce degenerate meshes. The
    # SurfaceMesh generated here persists through the later delete.
    for element in POLYHEDRA_ELEMENTS:
        if element not in present_elements:
            continue
        append_polyhedra(
            pipeline,
            element_color(element),
            selection_seeder=_make_b_neighbor_selector({element}),
            transparency=POLYHEDRA_TRANSPARENCY,
        )

    append_element_colors(pipeline, element_type_ids)

    # Isolate: keep only the target O and its first-neighbor cations.
    pipeline.modifiers.append(_make_environment_selector())
    pipeline.modifiers.append(DeleteSelectedModifier())

    # ======= Set up visual elements ======= #

    data = pipeline.compute()
    data.particles.vis.scaling = 0.5
    data.cell.vis.enabled = False

    # 22.5° tilt about X — matches the GUI viewpoint.
    pipeline.add_to_scene(rotation=[math.pi / 8, 0.0, 0.0])

    # ========== Set up rendering ========== #

    # Type.Top is the built-in straight-down-Z orthographic preset; combined
    # with the 22.5° scene rotation above it yields the slight forward tilt.
    vp = Viewport(type=Viewport.Type.Top)
    vp.zoom_all(size=size)

    # Same renderer settings as render_unit_cell.py, so the two Fig. 1
    # panels share one material and lighting style.
    renderer = OSPRayRenderer(
        samples_per_pixel=32,
        ambient_brightness=1.2,
        material_type="principled",
    )

    vp.render_image(filename=output_path, size=size, renderer=renderer)


def main():
    # Renders the panel that figure1.py drops in as Fig. 1b. TARGET_O_FRAC
    # is tied to the O3_3_1 site label in this specific CIF.
    render_o_site(
        "data/catio3_unit_cell.cif",
        "panel_1b_local_environment.png",
        size=(2000, 2000),   # ~5.4 cm panel at 600 dpi needs >= 1280 px
    )


if __name__ == "__main__":
    main()
