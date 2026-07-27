"""Regenerate the DFT-derived figure data from the raw VASP outputs.

This script is the connection between the deposited VASP calculations (the
NOMAD data deposit; locally, ``vasp-calculations/``) and the committed CSVs in
``figures/data/`` that the figure scripts and manuscript tables consume:

  - ``dft_ev_local_env.csv``  -- oxygen-vacancy formation energies, one row per
    symmetry-distinct site (90 rows), full precision.
  - ``dft_ev_sites.csv``      -- the same energies rounded to 0.01 eV (the CFM
    fit input; see ``figures/cfm_model.py``).
  - ``dft_ehull.csv``         -- energies above the convex hull for the six bulk
    compositions (Fig. 2), via ``ehull.py``/``hull_data.py``.
  - the ``DFT`` column of ``dgnn_preds_pretrained.csv``/``dgnn_preds_finetuned.csv``
    -- the SCAN+U reference values in the dGNN parity data (model predictions
    in those files are fixed dGNN outputs and are left untouched).

Ev is computed as

    Ev = E(defective supercell) - E(bulk supercell) + (1/2) E_O2

with every supercell energy read from the last ``energy(sigma->0)`` of the
corresponding OUTCAR, and E_O2 taken from ``hull_data.py`` (single source of
truth, shared with the E_hull construction): the SCAN total energy of the O2
molecule from G. Sai Gautam and E. A. Carter, Phys. Rev. Materials 2, 095401
(2018), the reference used for the whole Ca-Ce-Ti-Mn-O energy set there.

The VASP tree may have either layout:
  - the local working layout, with one ``NN_final`` subdirectory per
    calculation (``<comp>/bulk_cell_relaxation/<NN>_final/OUTCAR.bz2|.xz``), or
  - the flattened NOMAD-deposit layout, with the output files directly in each
    calculation directory (``<comp>/bulk_cell_relaxation/OUTCAR.bz2|.xz``).
Vacancy directories may use either naming convention: the local
``VV_iOO_Ce-Mn_F`` or the deposit ``VV_Ce-Mn_F_OOO`` (zero-padded ``F``).

Usage (paths are anchored to this file; both arguments optional):

    python figure_data_from_vasp.py [--vasp-root PATH] [--data-dir PATH]

Prints the weighted per-composition Ev statistics of main-text Table
"CCTM Ev statistics" (mean, median, percentile ranks of 3.4/3.9 eV, percent in
the 3.4-3.9 eV target window) for cross-checking against the manuscript.
"""
from __future__ import annotations

import argparse
import bz2
import lzma
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hull_data import elements  # noqa: E402  (E_O2 single source of truth)

E_O2 = elements["O2"]  # -12.3101 eV; Sai Gautam & Carter, PRM 2, 095401 (2018)

# (composition, M, N, group) in the row order of dft_ev_local_env.csv.
# M = Ti-count offset from the charge-balanced (Mn = 2 Ce) composition,
# N = Ce atoms per 360-atom cell.
COMPOSITIONS = [
    ("Ca54Ce18Ti36Mn36O216", 0, 18, "series_A"),
    ("Ca51Ce21Ti30Mn42O216", 0, 21, "series_A"),
    ("Ca48Ce24Ti24Mn48O216", 0, 24, "intersection"),
    ("Ca45Ce27Ti18Mn54O216", 0, 27, "series_A"),
    ("Ca48Ce24Ti18Mn54O216", -6, 24, "series_B"),
    ("Ca48Ce24Ti30Mn42O216", 6, 24, "series_B"),
]

FORMULA_RE = re.compile(r"Ca(\d+)Ce(\d+)Ti(\d+)Mn(\d+)O216")
SITE_RE = re.compile(r"(\d+)_i(\d+)_(\d)-(\d)_(\d+)")
SITE_RE_DEPOSIT = re.compile(r"(\d+)_(\d)-(\d)_(\d+)_(\d+)")
ENERGY_RE = re.compile(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)")


def final_energy(calc_dir: Path) -> float:
    """Last energy(sigma->0) of the calculation's OUTCAR (either layout)."""
    for candidate in (calc_dir / "OUTCAR.bz2", calc_dir / "OUTCAR.xz",
                      calc_dir / "OUTCAR"):
        if candidate.is_file():
            outcar = candidate
            break
    else:
        finals = sorted(calc_dir.glob("*_final"))
        if len(finals) != 1:
            raise FileNotFoundError(
                f"{calc_dir}: expected an OUTCAR or exactly one *_final subdirectory")
        return final_energy(finals[0])
    opener = {".bz2": bz2.open, ".xz": lzma.open}.get(outcar.suffix, open)
    energy = None
    with opener(outcar, "rt", errors="replace") as fh:
        for line in fh:
            match = ENERGY_RE.search(line)
            if match:
                energy = float(match.group(1))
    if energy is None:
        raise ValueError(f"no energy(sigma->0) in {outcar}")
    return energy


def site_rows(comp_dir: Path) -> list[dict]:
    """One dict per vacancy directory: site number, k_Ce, k_Mn, multiplicity."""
    vac_root = comp_dir / "vacancies_geometry_relaxation"
    rows = []
    for site_dir in sorted(vac_root.iterdir()):
        match = SITE_RE.fullmatch(site_dir.name)
        if match:
            number, _index, k_ce, k_mn, count = (int(g) for g in match.groups())
        else:
            match = SITE_RE_DEPOSIT.fullmatch(site_dir.name)
            if not match:
                continue
            number, k_ce, k_mn, count, _index = (int(g) for g in match.groups())
        rows.append({"dir": site_dir, "site": number, "k_ce": k_ce,
                     "k_mn": k_mn, "count": count})
    if len(rows) != 15:
        raise ValueError(f"{vac_root}: expected 15 vacancy directories, found {len(rows)}")
    return sorted(rows, key=lambda r: r["site"])


def weighted_stats(ev: np.ndarray, weights: np.ndarray) -> dict:
    """Multiplicity-weighted statistics for one composition (216 oxygen sites)."""
    order = np.argsort(ev)
    ev_sorted, w_sorted = ev[order], weights[order]
    cum = np.cumsum(w_sorted)
    median = ev_sorted[np.searchsorted(cum, cum[-1] / 2.0)]
    total = weights.sum()
    return {
        "mean": float(np.average(ev, weights=weights)),
        "median": float(median),
        "p34": 100.0 * weights[ev <= 3.4].sum() / total,
        "p39": 100.0 * weights[ev <= 3.9].sum() / total,
        "in_window": 100.0 * weights[(ev >= 3.4) & (ev <= 3.9)].sum() / total,
    }


def refresh_preds_dft(preds_path: Path, ev_table: pd.DataFrame) -> None:
    """Point the DFT column of a dGNN parity CSV at the recomputed Ev values.

    Rows are matched by composition and by proximity to the recomputed Ev;
    because the E_O2 revision shifts every Ev by the same 0.1825 meV, each old
    DFT value sits within a few 1e-4 eV of exactly one new value (the smallest
    Ev separation within a composition is 0.26 meV, checked below).
    """
    preds = pd.read_csv(preds_path)
    new_dft = []
    for tag, old in zip(preds["tag"], preds["DFT"]):
        comp = tag.split("-")[0]
        candidates = ev_table.loc[ev_table["composition"] == comp, "Ev"].to_numpy()
        residuals = np.abs(candidates - old)
        best = np.argsort(residuals)
        if residuals[best[0]] > 5e-4 or residuals[best[1]] < 2 * residuals[best[0]]:
            raise ValueError(f"{preds_path.name}: ambiguous match for {tag}")
        new_dft.append(candidates[best[0]])
    preds["DFT"] = new_dft
    preds.to_csv(preds_path, index=False)
    print(f"  refreshed DFT column: {preds_path.name} ({len(preds)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vasp-root", type=Path,
                        default=SCRIPT_DIR.parent / "vasp-calculations",
                        help="root of the VASP tree (local layout or NOMAD deposit layout)")
    parser.add_argument("--data-dir", type=Path,
                        default=SCRIPT_DIR.parent / "figures" / "data",
                        help="output directory (the repository's figures/data)")
    args = parser.parse_args()

    ev_rows, ehull_rows = [], []
    for comp, m, n, group in COMPOSITIONS:
        n_ca, n_ce, n_ti, n_mn = (int(g) for g in FORMULA_RE.fullmatch(comp).groups())
        x_ce, x_mn = n_ce / (n_ca + n_ce), n_mn / (n_ti + n_mn)
        e_bulk = final_energy(args.vasp_root / comp / "bulk_cell_relaxation")
        print(f"{comp}: E_bulk = {e_bulk:.6f} eV")
        ehull_rows.append({"M": m, "N": n, "composition": comp, "x_Ce": x_ce,
                           "x_Mn": x_mn, "E_bulk": e_bulk, "group": group})
        for row in site_rows(args.vasp_root / comp):
            e_def = final_energy(row["dir"])
            ev_rows.append({
                "Composition": f"M={m}, N={n}", "M": m, "N": n,
                "XCe": x_ce, "XMn": x_mn,
                "xCe": row["k_ce"] / 4.0, "xMn": row["k_mn"] / 2.0,
                "Count": row["count"], "Ev": e_def - e_bulk + E_O2 / 2.0,
                "group": group, "composition": comp,
                "NCe": n_ce, "NMn": n_mn,
            })

    ev = pd.DataFrame(ev_rows)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    ev.drop(columns=["composition", "NCe", "NMn"]).to_csv(
        args.data_dir / "dft_ev_local_env.csv", index=False)
    sites = pd.DataFrame({
        "Composition": ev["Composition"], "NA": 72, "NCe": ev["NCe"],
        "NMn": ev["NMn"], "xCe": ev["xCe"], "xMn": ev["xMn"],
        "Ev": ev["Ev"].round(2),
    })
    sites.to_csv(args.data_dir / "dft_ev_sites.csv", index=False)
    print(f"wrote dft_ev_local_env.csv and dft_ev_sites.csv ({len(ev)} sites, "
          f"E_O2 = {E_O2} eV)")

    for preds_name in ("dgnn_preds_pretrained.csv", "dgnn_preds_finetuned.csv"):
        preds_path = args.data_dir / preds_name
        if preds_path.is_file():
            refresh_preds_dft(preds_path, ev)

    # Overall parity metrics derive entirely from the preds files (MAE and
    # Pearson r^2 of the ensemble-mean prediction against the DFT reference),
    # so they are regenerated here.  The per-fold metrics in dgnn_cv_folds.csv
    # are NOT regenerable from committed data (fold membership is not
    # recorded); they retain the upstream E_O2 reference, a < 0.2 meV effect.
    base_path = args.data_dir / "dgnn_preds_pretrained.csv"
    ft_path = args.data_dir / "dgnn_preds_finetuned.csv"
    if base_path.is_file() and ft_path.is_file():
        base, ft = pd.read_csv(base_path), pd.read_csv(ft_path)
        overall = pd.DataFrame([{
            "all_FT_MAE": float(np.mean(np.abs(ft["avgDefCv"] - ft["DFT"]))),
            "all_FT_R2": float(np.corrcoef(ft["avgDefCv"], ft["DFT"])[0, 1] ** 2),
            "all_base_MAE": float(np.mean(np.abs(base["avgDefCv"] - base["DFT"]))),
            "all_base_R2": float(np.corrcoef(base["avgDefCv"], base["DFT"])[0, 1] ** 2),
        }])
        overall.to_csv(args.data_dir / "dgnn_cv_overall.csv", index=False)
        print("  recomputed dgnn_cv_overall.csv")

    try:
        from ehull import phase_diagram_with
    except ImportError as exc:
        print(f"skipping dft_ehull.csv (pymatgen unavailable: {exc})")
    else:
        for row in ehull_rows:
            phase_diagram, cctm = phase_diagram_with(row["composition"], row.pop("E_bulk"))
            _, e_hull = phase_diagram.get_decomp_and_e_above_hull(cctm)
            row["ehull_in_meV"] = 1000.0 * e_hull
        ehull = pd.DataFrame(ehull_rows).sort_values(["M", "N"])
        ehull = ehull[["M", "N", "composition", "x_Ce", "x_Mn", "ehull_in_meV", "group"]]
        ehull.to_csv(args.data_dir / "dft_ehull.csv", index=False)
        print("wrote dft_ehull.csv")

    print("\nWeighted per-composition Ev statistics (main-text Table cross-check):")
    print("composition            mean  median  p(<=3.4)  p(<=3.9)  in-window")
    for comp, m, n, _ in COMPOSITIONS:
        sub = ev[ev["composition"] == comp]
        s = weighted_stats(sub["Ev"].to_numpy(), sub["Count"].to_numpy(float))
        print(f"{comp}  {s['mean']:.2f}  {s['median']:.2f}    "
              f"{s['p34']:5.1f}     {s['p39']:5.1f}      {s['in_window']:5.1f}")


if __name__ == "__main__":
    main()
