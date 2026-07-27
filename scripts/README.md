# Deposited analysis scripts

Self-contained copies of the analysis scripts behind the published results, adapted to run against the data in this repository (the originals ran inside a separate plotting project through machine-specific paths). Each script anchors its paths to its own location, so all of them can be run from any working directory.

| File | Role | Manuscript artifact |
| --- | --- | --- |
| `INCAR` | Representative VASP input for the SCAN+*U* relaxations (pristine cells, `ISIF = 3`; defective cells used `ISIF = 2`, see Methods). | Methods |
| `figure_data_from_vasp.py` | Regenerates the DFT-derived figure data from the raw VASP outputs (the NOMAD deposit): oxygen-vacancy formation energies (`dft_ev_local_env.csv`, `dft_ev_sites.csv`), hull energies (`dft_ehull.csv`), the DFT reference column of the dGNN parity data, and the overall parity metrics (`dgnn_cv_overall.csv`). | *E*<sub>v</sub> statistics table; Figs. 3–5 inputs |
| `dgnn_delta_grid.py` | Converts the fine-tuned dGNN predictions for the sampled oxygen environments (`figures/data/dgnn_screen_finetuned.csv`) into the redox-capacity grid `figures/data/dgnn_delta_grid.csv`. | Fig. 5b |
| `tga_preprocess.py` | Extracts the three figure columns (time, temperature, delta) from the raw TGA workbooks in `figures/data/raw/tga/` and writes `figures/data/tga_*.csv`, reproducing the CSVs in `figures/data/` byte for byte. | ESI TGA figure |
| `hull_data.py` | SCAN+*U* total energies (eV) of the elemental, binary, and ternary reference phases used in the convex-hull analysis, from prior published work (G. Sai Gautam and E. A. Carter, *Phys. Rev. Mater.*, 2018, **2**, 095401; O. Y. Long, G. Sai Gautam and E. A. Carter, *Phys. Rev. Mater.*, 2020, **4**, 045401; G. Sai Gautam, E. B. Stechel and E. A. Carter, *Chem. Mater.*, 2020, **32**, 9964). Its `O2` entry is the O₂ reference energy for all *E*<sub>v</sub> values. | Fig. 2; all *E*<sub>v</sub> |
| `ehull.py` | Builds the pymatgen phase diagram from `hull_data.py` and reports the decomposition and energy above the hull for a given cell composition and DFT total energy. | Fig. 2 |

## Provenance chain

Every DFT-derived number in the manuscript can be followed from the raw VASP outputs to the published artifact:

1. **Raw calculations** -- the NOMAD deposit, DOI 10.17172/nomad.6d7e-hvb1 (one directory per calculation: 6 bulk relaxations + 90 oxygen-vacancy relaxations).
2. **`figure_data_from_vasp.py`** extracts the final `energy(sigma->0)` of every run and writes the vacancy-formation-energy and hull-energy CSVs in `figures/data/`, using `E_O2` from `hull_data.py` (single source of truth).
3. **`figures/cfm_model.py`** re-fits the CFM on `dft_ev_sites.csv` and computes the Table 3 CFM and SCAN+*U* Δδ columns; `dgnn_delta_grid.py` produces the dGNN grid.
4. **`figures/figure*.py`** render the PNGs in `figures/plots/` from the CSVs (see `figures/README.md` for the per-figure map).

The experimental chain is shorter: `tga_preprocess.py` converts the as-received TGA workbooks into the `tga_*.csv` files; the XRD CSVs and LSFR JSON files are themselves the as-received data (see the provenance tables in `figures/README.md`).

Not regenerable from the data included here: the dGNN model outputs themselves (the prediction columns of `dgnn_preds_*.csv` and `dgnn_screen_finetuned.csv`; the dGNN is external), the per-fold metrics in `dgnn_cv_folds.csv` (fold membership is not recorded), and the 2×2×2 *k*-grid DOS check (its run is not in the deposit; see the deposit README completeness notes).

POTCAR files are not redistributable under the VASP license; the PAW potential identities are listed in the Methods.

## Usage

```bash
python scripts/figure_data_from_vasp.py    # from the raw VASP tree to figures/data/
python scripts/dgnn_delta_grid.py
python scripts/tga_preprocess.py
python scripts/ehull.py Ca48Ce24Ti18Mn54O216 <total energy in eV>
```
