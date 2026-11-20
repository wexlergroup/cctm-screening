# cctm-screening

The processed data, the scripts that generate all figures, and the figures for:

> Manish Kumar, Natalia Ali, Matthew D. Witman, Shang Zhai, James E. Miller, Ivan Ermanoski, Ellen B. Stechel, and Robert B. Wexler, *Local B-site chemistry controls oxygen-vacancy energetics in Ca–Ce–Ti–Mn perovskites for thermochemical hydrogen production*, *J. Mater. Chem. A*, 2026, accepted (preprint: [arXiv:2607.28752](https://arxiv.org/abs/2607.28752)).

The study maps oxygen-vacancy formation energetics in Ca–Ce–Ti–Mn (CCTM) perovskites for two-step solar thermochemical water splitting, combining SCAN+*U* density-functional-theory calculations, two complementary fitted models (a linear crystal-feature model, CFM, and a defect graph neural network, dGNN), and experimental synthesis, X-ray diffraction, thermogravimetric analysis, and stagnation-flow-reactor cycling.

## Repository layout

| Path | Contents |
| --- | --- |
| `figures/` | One standalone Python script per figure, with the shared style in `fig_style.py`. See `figures/README.md` for the per-figure map. |
| `figures/plots/` | The 600-dpi PNGs exactly as embedded in the manuscript, one per main-text and ESI figure; regenerating them with the scripts reproduces these files byte for byte under the conditions given in Reproducing the figures. |
| `figures/data/` | Processed data consumed by the figure scripts: DFT-derived CSVs (`dft_*`), fitted-model outputs (`cfm_*`, `dgnn_*`), and experimental measurements (`tga_*`, `xrd_*`, `lsfr_*`). Provenance tables in `figures/README.md`. |
| `figures/data/raw/tga/` | As-received thermogravimetric-analysis workbooks (verbatim instrument-export filenames). |
| `figures/data/raw/xrd/` | As-received X-ray diffraction scans of the as-synthesized powders and of the samples recovered after cycling (two-column 2θ/intensity files). |
| `scripts/` | Path-portable analysis scripts connecting the raw data to the processed CSVs, plus a representative VASP `INCAR`. See `scripts/README.md` for the full provenance chain. |
| `cfm-settings/features.csv` | Atomic descriptor table for the crystal-feature model: one row per cation species (22 species), with values from [R. B. Wexler, G. Sai Gautam, E. B. Stechel and E. A. Carter, *J. Am. Chem. Soc.*, 2021, **143**, 13212](https://doi.org/10.1021/jacs.1c05570). See `cfm-settings/README.md` for the schema. |

## Provenance chain

Every DFT-derived number in the manuscript can be followed from the raw VASP outputs to the published artifact:

1. Raw VASP calculations: NOMAD repository, DOI [10.17172/nomad.6d7e-hvb1](https://doi.org/10.17172/nomad.6d7e-hvb1) (6 bulk relaxations, 90 oxygen-vacancy relaxations, and 2 static runs on a 2×2×2 *k*-grid).
2. `scripts/figure_data_from_vasp.py` extracts the final energies and writes the vacancy-formation-energy and hull-energy CSVs in `figures/data/`.
3. `figures/cfm_model.py` re-fits the CFM and computes the Table 3 columns; `scripts/dgnn_delta_grid.py` produces the dGNN redox-capacity grid.
4. `figures/figure*.py` render the PNGs in `figures/plots/` from the CSVs.

The experimental chain is shorter: `scripts/tga_preprocess.py` converts the as-received TGA workbooks into the `tga_*.csv` files byte for byte; the XRD CSVs, the post-cycling XRD scans in `figures/data/raw/xrd/`, and the stagnation-flow-reactor JSON files are themselves the as-received data.

Not regenerable from this repository: the dGNN prediction columns (the model is external) and the per-fold metrics in `dgnn_cv_folds.csv`.

## Reproducing the figures

```bash
pip install -r requirements.txt
cd figures
python figure3.py        # writes plots/figure3.png
```

Each script reads from `figures/data/` and writes one PNG to `figures/plots/`; run scripts from inside `figures/`. Regeneration is deterministic: with the package versions pinned in `requirements.txt` (they install on Python 3.10 to 3.13) and matplotlib's default macOS backend (`macosx`) on a machine where the Arial font resolves, an unchanged script reproduces the PNG in `figures/plots/` byte for byte. Pillow is pinned because it writes the PNG files: on macOS, Pillow 12.3.0 encodes the same pixels into different bytes. With the `Agg` backend (the default on a headless or Linux machine) the output is visually equivalent, but text positions and image dimensions shift by a few pixels, and without Arial matplotlib substitutes another font. The Fig. 1 structure panels additionally require `pip install ovito`; the rendered panels are included, so OVITO is needed only to re-render them.

## Data availability

- Processed data and figure scripts: this repository. Each release is archived at Zenodo, and the DOI of the archived version is given in the Data availability statement of the article.
- Raw VASP inputs and outputs: NOMAD, DOI [10.17172/nomad.6d7e-hvb1](https://doi.org/10.17172/nomad.6d7e-hvb1).

## Licensing

The data and figures (`figures/data/`, `figures/plots/`, the two Fig. 1 structure panels in `figures/`, and `cfm-settings/features.csv`) are available under the Creative Commons Attribution 4.0 International license (CC BY 4.0); see `LICENSE-DATA`. Reuse of them requires attribution to the article cited above. The figure and analysis code, the READMEs, and the configuration files are available under the MIT license; see `LICENSE`. The raw density-functional-theory inputs and outputs are deposited separately on NOMAD and carry that repository's terms.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Fuel Cell Technologies Office (FCTO), Award Number DE-EE0010733, through the HydroGEN Advanced Water Splitting Materials Consortium.
