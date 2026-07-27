# Atomic descriptor table for the crystal-feature model

`features.csv` holds one row per cation species (element + oxidation state). The 22 species cover the A-site cations Ba²⁺, Ca²⁺, Ce³⁺/Ce⁴⁺, La³⁺, and Sr²⁺ and the B-site cations Co²⁺/Co³⁺/Co⁴⁺, Cr³⁺/Cr⁴⁺, Fe³⁺/Fe⁴⁺, Mn³⁺/Mn⁴⁺, Ni²⁺/Ni³⁺/Ni⁴⁺, Ti³⁺/Ti⁴⁺, and V³⁺/V⁴⁺. The values are those of R. B. Wexler, G. Sai Gautam, E. B. Stechel and E. A. Carter, *J. Am. Chem. Soc.*, 2021, **143**, 13212, DOI [10.1021/jacs.1c05570](https://doi.org/10.1021/jacs.1c05570), which defines every quantity; the table was normalized from that work's per-material feature table (one row per perovskite, species descriptors repeated), with every retained value preserved exactly.

The crystal-feature model in this repository (`figures/cfm_model.py`) reads four columns: `SCAN_CBDE` and `SCAN_Vred` (SCAN-computed descriptors) and `exp_CBDE` and `exp_Vred` (experimental counterparts). The remaining columns document how those descriptors were derived.

## Columns

| Column | Meaning |
| --- | --- |
| `species`, `element`, `ox_state` | Species key, e.g. `Mn4` = Mn in oxidation state +4. |
| `form`, `E0_MOx`, `CN` | Reference binary oxide, its SCAN total energy (eV per formula unit), and the cation coordination number in it. |
| `E0_atom`, `E0_O_atom` | SCAN energies of the isolated metal and oxygen atoms (eV). |
| `n_met`, `n_O` | Metal and oxygen counts per reference-oxide formula unit. |
| `SCAN_Ecoh`, `exp_Ecoh` | Cohesive energy of the reference oxide (eV), SCAN and experimental. |
| `SCAN_CBDE`, `exp_CBDE` | Crystal bond dissociation energy (eV), SCAN and experimental; the *E*<sub>b</sub> descriptor. |
| `red`, `form_red`, `E0_red` | Oxidation state and phase of the reduced reference (metal or lower oxide) and its SCAN total energy (eV). |
| `n_met_red`, `n_O2`, `E0_O2` | Reduction-reaction stoichiometry and the O₂ reference energy (eV; the same value as `hull_data.py`). |
| `Ered`, `SCAN_Vred`, `exp_Vred` | Reduction-reaction energy (eV) and crystal reduction potentials (V), SCAN and experimental; the *V*<sub>r</sub> descriptor. |

Columns with no value for a species (for example, `exp_CBDE` for Co⁴⁺, Fe⁴⁺, and Ni⁴⁺) are empty: no experimental reference oxide exists for them.
