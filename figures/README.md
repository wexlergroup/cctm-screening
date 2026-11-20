# Figure pipeline

Every figure in the manuscript (`main.tex`) and the ESI (`esi.tex`) is produced by one standalone script in this directory, reading processed data from `data/` and writing a 600-dpi PNG to `plots/`. The PNGs in `plots/` are what the documents embed, so the paper compiles without running any Python; run a script only to change its figure, and the PNG it writes replaces the previous one.

Regeneration is deterministic: an unchanged script regenerates its PNG in `plots/` byte for byte with matplotlib's default macOS backend (`macosx`) on a machine where Arial resolves (check stderr for `findfont` warnings). With the `Agg` backend (the default on a headless or Linux machine) the output is visually equivalent, but text positions and image dimensions shift by a few pixels. All scripts run from inside `figures/`:

```bash
cd figures && python3 figure3.py
```

## Naming scheme

- `figure<N>.py` → `plots/figure<N>.png`: main-text figures, one script per figure, numbered as in the paper.
- `figureS_<topic>.py` → `plots/figureS_<topic>.png`: ESI figures, named by topic (ESI figure numbers shift with reordering; the `esi.tex` label is the identity).
- Data files: `<source>_<content>` with the source prefix `dft_`, `dgnn_`, `cfm_`, `tga_`, `xrd_`, or `lsfr_`. Raw instrument exports under `data/raw/` keep their verbatim as-received names.
- `fig_style.py` holds the shared EES rcParams, column widths, and model colors; importing it applies the canonical style, and scripts keep only figure-specific overrides locally.

## Figure map

| Figure | Script | Reads (data/) | Output (plots/) |
| --- | --- | --- | --- |
| Fig. 1 | `figure1.py` | `bulk_compositions.csv`; panel PNGs `panel_1a_bulk_structure.png`, `panel_1b_local_environment.png` (OVITO renders, see below) | `figure1.png` |
| Fig. 2 | `figure2.py` | `dft_ehull.csv` | `figure2.png` |
| Fig. 3 | `figure3.py` | `dft_ev_local_env.csv` | `figure3.png` |
| Fig. 4 | `figure4.py` | `dft_ev_local_env.csv` | `figure4.png` |
| Fig. 5 | `figure5.py` | `cfm_delta_grid.csv`, `dgnn_delta_grid.csv` | `figure5.png` |
| Fig. 6 | `figure6.py` | `lsfr_CCTM1771014.json`, `lsfr_CCTM2112.json` | `figure6.png` |
| ESI magnetic moments | `figureS_magmoments.py` | `dft_magmoments_bulk.csv` | `figureS_magmoments.png` |
| ESI DOS | `figureS_dos.py` | `dft_dos.csv` | `figureS_dos.png` |
| ESI CFM parity | `figureS_cfm_parity.py` | `dft_ev_sites.csv` | `figureS_cfm_parity.png` |
| ESI vacancy–vacancy interaction | `figureS_vv.py` | (analytic; no inputs) | `figureS_vv.png` |
| ESI XRD | `figureS_xrd.py` | `xrd_CCTM*.csv`, `cctm2112_sqs.cif` | `figureS_xrd.png` |
| ESI post-cycling XRD | `figureS_xrd_postcycling.py` | `raw/xrd/CCTM*.xy` | `figureS_xrd_postcycling.png` |
| ESI TGA | `figureS_tga.py` | `tga_CCTM*_protocol_B.csv` (the protocol-A CSVs are included as data but not plotted) | `figureS_tga.png` |
| ESI dGNN parity | `figureS_dgnn_parity.py` | `dgnn_preds_pretrained.csv`, `dgnn_preds_finetuned.csv`, `dgnn_cv_folds.csv`, `dgnn_cv_overall.csv` | `figureS_dgnn_parity.png` |
| ESI dGNN vs CFM distributions | `figureS_dgnn_vs_cfm.py` | `dgnn_screen_finetuned.csv` | `figureS_dgnn_vs_cfm.png` |
| ESI dGNN screening summary | `figureS_dgnn_summary.py` | `dgnn_screen_finetuned.csv` | `figureS_dgnn_summary.png` |
| ESI k-grid convergence | `figureS_bz_convergence.py` | `dft_dos.csv` | `figureS_bz_convergence.png` |

`cfm_model.py` is not a figure script: it is the canonical implementation of the crystal-feature model. It re-fits the CFM coefficients on `dft_ev_sites.csv` (descriptors from `../cfm-settings/features.csv`), prints the Table 3 CFM and SCAN+*U* columns, and writes `cfm_delta_grid.csv` for `figure5.py`.

The Fig. 1 structure panels are OVITO renders: `render_unit_cell.py` and `render_o_site.py` (helpers in `render_utils.py`, element colors in `style_params.py`) read `data/catio3_unit_cell.cif` and write the two `panel_1*.png` files at this directory's root, which `figure1.py` autocrops and composites. They need `pip install ovito` (deliberately outside `requirements.txt`; the panel PNGs are included, so OVITO is required only to re-render them).

## Data provenance (`data/`)

Computational chain (see `../scripts/README.md` for the full description):

| File | Produced by | Upstream source |
| --- | --- | --- |
| `dft_ev_local_env.csv`, `dft_ev_sites.csv`, `dft_ehull.csv`, `dgnn_cv_overall.csv`, DFT column of `dgnn_preds_*.csv` | `../scripts/figure_data_from_vasp.py` | raw VASP outputs (NOMAD deposit DOI [10.17172/nomad.6d7e-hvb1](https://doi.org/10.17172/nomad.6d7e-hvb1)) |
| `cfm_delta_grid.csv` | `cfm_model.py` | `dft_ev_sites.csv` + `../cfm-settings/features.csv` |
| `dgnn_delta_grid.csv` | `../scripts/dgnn_delta_grid.py` | `dgnn_screen_finetuned.csv` |
| `dft_dos.csv` | extracted from the deposit's `vasprun.xml` files (Γ-only and 2×2×2 *k*-grids; `kgrid` column) | NOMAD deposit |
| `dft_magmoments_bulk.csv`, `dft_magmoments_vacancy.csv` | PAW-sphere columns (`mu_PAW`, `dmu_PAW`) extracted from the deposit's OUTCARs (unsigned moments); Bader columns (`mu_Bader`, `dmu_Bader`) from separate static Bader-analysis runs; `dft_magmoments_vacancy.csv` is a data extra read by no figure script | NOMAD deposit for the PAW-sphere columns; the Bader runs are not part of the deposit, whose README lists the INCAR changes that regenerate them |
| `dgnn_preds_pretrained.csv`, `dgnn_preds_finetuned.csv`, `dgnn_screen_finetuned.csv`, `dgnn_cv_folds.csv` (prediction columns) | fixed dGNN model outputs, not regenerable from the data included here | external dGNN training (T. C. Douglas *et al.*, *Mater. Horiz.*, 2026, **13**, DOI [10.1039/D5MH01566A](https://doi.org/10.1039/D5MH01566A)) |
| `bulk_compositions.csv` | composition design table for the sampled bulks | defined in this work |
| `cctm2112_sqs.cif` | CCTM2112 SQS supercell (calculated XRD reference) | this work's SQS construction |
| `catio3_unit_cell.cif` | CaTiO₃ unit cell used for the Fig. 1 structure renders | Materials Project CaTiO₃, reoriented |

Experimental chain (samples synthesized at ASU; measurements as noted):

| File | Produced by | Upstream source |
| --- | --- | --- |
| `tga_CCTM*_protocol_{A,B}.csv` | `../scripts/tga_preprocess.py` (byte-for-byte reproducible) | raw TGA workbooks in `data/raw/tga/` (as-received instrument exports, verbatim filenames) |
| `xrd_CCTM*.csv` | as received (two-column 2θ/intensity exports from ASU) | ASU diffractometer exports; raw scans not held in this repository |
| `raw/xrd/CCTM*_{fresh,postLSFR,postTGA}.xy` | as received (two-column 2θ/intensity exports from ASU, 28 Oct 2026; `CCTM1771014_postLSFR.xy` renamed only to remove a space) | ASU diffractometer scans of the as-synthesized powders and of the samples recovered after LSFR or TGA cycling |
| `lsfr_CCTM*.json` | as received (structured export from the Sandia laser-heated stagnation flow reactor) | SNL LSFR capability node; these files are the raw cycling data |

`data/raw/tga/CCTM TGA + LSFR Data Updated.xlsx` is the collaborators' summary compilation. Its `Initial TGA Data Summary` sheet holds the Δ*δ* values reported in the TGA columns of Table 3, and its `LSFR Data Summary` sheet holds the per-step LSFR values; it is not read by any script.
