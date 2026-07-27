# cctm-screening

The processed data, the scripts that generate all figures, and the figures for:

> Manish Kumar, Natalia Ali, Matthew D. Witman, Shang Zhai, James E. Miller, Ivan Ermanoski, Ellen B. Stechel, and Robert B. Wexler, *Local B-site chemistry controls oxygen-vacancy energetics in Ca–Ce–Ti–Mn perovskites for thermochemical hydrogen production* (citation to be updated upon publication).

The study maps oxygen-vacancy formation energetics in Ca–Ce–Ti–Mn (CCTM) perovskites for two-step solar thermochemical water splitting, combining SCAN+*U* density-functional-theory calculations, two complementary fitted models (a linear crystal-feature model, CFM, and a defect graph neural network, dGNN), and experimental synthesis, X-ray diffraction, thermogravimetric analysis, and stagnation-flow-reactor cycling.

## Repository layout

| Path | Contents |
| --- | --- |
| `figures/` | One standalone Python script per figure, with the shared style in `fig_style.py`. See `figures/README.md` for the per-figure map. |
| `figures/plots/` | The 600-dpi PNGs exactly as embedded in the manuscript, one per main-text and ESI figure; regenerating them with the scripts reproduces these files byte for byte. |
| `figures/data/` | Processed data consumed by the figure scripts: DFT-derived CSVs (`dft_*`), fitted-model outputs (`cfm_*`, `dgnn_*`), and experimental measurements (`tga_*`, `xrd_*`, `lsfr_*`). Provenance tables in `figures/README.md`. |
| `figures/data/raw/tga/` | As-received thermogravimetric-analysis workbooks (verbatim instrument-export filenames). |
| `scripts/` | Path-portable analysis scripts connecting the raw data to the processed CSVs, plus a representative VASP `INCAR`. See `scripts/README.md` for the full provenance chain. |
| `cfm-settings/features.csv` | Atomic descriptor table for the crystal-feature model: one row per cation species (22 species), with values from [R. B. Wexler, G. Sai Gautam, E. B. Stechel and E. A. Carter, *J. Am. Chem. Soc.*, 2021, **143**, 13212](https://doi.org/10.1021/jacs.1c05570). See `cfm-settings/README.md` for the schema. |

## Provenance chain

Every DFT-derived number in the manuscript can be followed from the raw VASP outputs to the published artifact:

1. Raw VASP calculations: NOMAD repository, DOI [10.17172/nomad.6d7e-hvb1](https://doi.org/10.17172/nomad.6d7e-hvb1) (6 bulk relaxations + 90 oxygen-vacancy relaxations).
2. `scripts/figure_data_from_vasp.py` extracts the final energies and writes the vacancy-formation-energy and hull-energy CSVs in `figures/data/`.
3. `figures/cfm_model.py` re-fits the CFM and computes the Table 3 columns; `scripts/dgnn_delta_grid.py` produces the dGNN redox-capacity grid.
4. `figures/figure*.py` render the PNGs in `figures/plots/` from the CSVs.

The experimental chain is shorter: `scripts/tga_preprocess.py` converts the as-received TGA workbooks into the `tga_*.csv` files byte for byte; the XRD CSVs and the stagnation-flow-reactor JSON files are themselves the as-received data.

Not regenerable from this repository: the dGNN prediction columns (the model is external) and the per-fold metrics in `dgnn_cv_folds.csv`.

## Reproducing the figures

```bash
pip install -r requirements.txt
cd figures
python figure3.py        # writes plots/figure3.png
```

Each script reads from `figures/data/` and writes one PNG to `figures/plots/`; run scripts from inside `figures/`. Regeneration is deterministic: on a machine where the Arial font resolves, an unchanged script reproduces the PNG in `figures/plots/` byte for byte. The Fig. 1 structure panels additionally require `pip install ovito`; the rendered panels are included, so OVITO is needed only to re-render them.

## Data availability

- Processed data and figure scripts: this repository, archived at Zenodo (DOI to be assigned on deposition).
- Raw VASP inputs and outputs: NOMAD, DOI [10.17172/nomad.6d7e-hvb1](https://doi.org/10.17172/nomad.6d7e-hvb1).

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Fuel Cell Technologies Office (FCTO), Award Number DE-EE0010733, through the HydroGEN Advanced Water Splitting Materials Consortium.
