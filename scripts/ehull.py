"""Energy above the convex hull for a CCTM composition.

Builds the Ca-Ce-Ti-Mn-O phase diagram from the SCAN+U reference energies in
``hull_data.py`` (elemental, binary, and ternary phases) plus the CCTM entry
under test, then reports its decomposition and energy above the hull (the
quantity plotted in Fig. 2, converted there to meV/atom).

This is a self-contained copy of the hull-analysis core used for the published
E_hull values; the original's machine-specific OUTCAR harvesting has been
replaced by command-line arguments.

Usage:

    python ehull.py Ca48Ce24Ti18Mn54O216 <total energy in eV>

where the energy is the VASP total energy of the relaxed cell with that
composition.
"""
from __future__ import annotations

import argparse

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from hull_data import elements, binaries, ternaries


def phase_diagram_with(compos: str, energy: float) -> tuple[PhaseDiagram, PDEntry]:
    """Phase diagram over all reference phases plus the CCTM entry under test.

    Polymorph-tagged keys in hull_data.py (e.g. ``Ca2V2O7_P1``) keep only the
    formula part before the underscore.
    """
    entries = []
    phases = {**elements, **binaries, **ternaries}
    for key, value in phases.items():
        entries.append(PDEntry(Composition(key.split("_")[0]), energy=value))
    cctm = PDEntry(Composition(compos), energy=energy)
    entries.append(cctm)
    return PhaseDiagram(entries=entries), cctm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decomposition and energy above the hull for a CCTM cell.")
    parser.add_argument("composition", help='cell composition, e.g. "Ca48Ce24Ti18Mn54O216"')
    parser.add_argument("energy", type=float, help="VASP total energy of the relaxed cell (eV)")
    args = parser.parse_args()

    phase_diagram, cctm = phase_diagram_with(args.composition, args.energy)
    decomp, e_hull = phase_diagram.get_decomp_and_e_above_hull(cctm)
    print("Decomposition:")
    for entry, fraction in decomp.items():
        print(f"  {entry.composition.reduced_formula}: {fraction:.4f}")
    print(f"E_hull = {e_hull:.6f} eV/atom = {1000.0 * e_hull:.2f} meV/atom")


if __name__ == "__main__":
    main()
