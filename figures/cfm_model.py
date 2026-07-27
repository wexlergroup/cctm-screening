"""Crystal-feature model (CFM): coefficient fit and Delta-delta screening.

Single source of truth for the CFM used in the main text (Table 3 CFM column and
Fig. 5a) and the ESI parity figure. Everything is derived from committed data:

  - Atomic descriptor values (Eb = cation bond-dissociation energy, Vr = crystal
    reduction potential) for the five species are read from
    ``cfm-settings/features.csv`` (columns CBDE/exp_CBDE and SCAN_Vred/exp_Vred).
  - The two coefficients (a_Eb, b_Vr) and the intercept are *re-fit* here by Huber
    regression on the 90 SCAN+U site energies in ``data/dft_ev_sites.csv``, using
    the same fractional-Mn local-environment features as the parity figure. They
    are not hard-coded, so the fit and the screening cannot drift apart.

Model (main-text Methods; ESI CFM section):
  - Two descriptors only: <Eb> and <Vr>. The energy above the convex hull is *not*
    a descriptor (its coefficient is statistically unresolved; see ESI subset table).
  - Mn carries a composition-dependent fractional valence set by *bulk* charge
    balance, n_Mn = 4 - X_Ce/X_Mn, and its descriptor values are obtained by
    linearly interpolating the Mn(III) (Mn2O3) and Mn(IV) (MnO2) references.
  - For a bulk (X_Ce, X_Mn), the 15 symmetry-distinct O nearest-neighbour
    environments (k_Ce in 0..4 A-site, k_Mn in 0..2 B-site) are binomially weighted;
    each environment's Ev is mapped to a vacancy occupancy by Fermi-Dirac statistics
    at the oxygen chemical potential, and delta = 3 * <occupancy>.

Run from inside ``figures/`` (or anywhere -- paths are anchored to this file):

    python cfm_model.py            # re-fit, print coefficients + Table 3, write the grid CSV

Writes ``data/cfm_delta_grid.csv`` (consumed by figure5.py).
"""
from __future__ import annotations

import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEATURES_FILE = ROOT / "cfm-settings" / "features.csv"
SITES_FILE = HERE / "data" / "dft_ev_sites.csv"
GRID_CSV = HERE / "data" / "cfm_delta_grid.csv"

# Cycling conditions (main-text Methods / ESI worked example)
T_RED, PO2_RED = 1623.15, 1e-4          # 1350 C, pO2 = 1e-4 bar
T_OX, PH2_OVER_PH2O = 1123.15, 1e-3     # 850 C, pH2/pH2O = 1e-3
GRID_POINTS_PER_AXIS = 10

# Which descriptor-table row supplies each species. The table is keyed by
# (element, ox_state); the reduction couple is recorded for the reader, and its
# numeric value is the row's Vred column.
#   key_eb, key_vr, element, oxidation state, couple
SPECIES_SPEC = [
    ("Ca2", "Ca2_0", "Ca", 2, "Ca2+ -> Ca0"),
    ("Ce3", "Ce3_0", "Ce", 3, "Ce3+ -> Ce0"),
    ("Ti4", "Ti4_3", "Ti", 4, "Ti4+ -> Ti3+"),
    ("Mn4", "Mn4_3", "Mn", 4, "Mn4+ -> Mn3+"),
    ("Mn3", "Mn3_2", "Mn", 3, "Mn3+ -> Mn2+"),
]


def load_species(features_file: Path = FEATURES_FILE) -> dict[str, dict[str, float]]:
    """Extract Eb (DFT/EXP) and Vr (DFT/EXP) for the five CCTM species.

    The descriptor table holds one row per cation species, keyed by
    (element, ox_state). Returns a dict with keys ``EB_DFT``, ``EB_EXP``,
    ``VR_DFT``, ``VR_EXP``.
    """
    df = pd.read_csv(features_file)
    EB_DFT, EB_EXP, VR_DFT, VR_EXP = {}, {}, {}, {}
    for key_eb, key_vr, elem, ox, _couple in SPECIES_SPEC:
        sub = df[(df["element"] == elem) & (df["ox_state"] == ox)]
        if len(sub) != 1:
            raise ValueError(
                f"features.csv must have exactly one {elem}{ox} row, found {len(sub)}")
        row = sub.iloc[0]
        EB_DFT[key_eb] = float(row["SCAN_CBDE"])
        EB_EXP[key_eb] = float(row["exp_CBDE"])
        VR_DFT[key_vr] = float(row["SCAN_Vred"])
        VR_EXP[key_vr] = float(row["exp_Vred"])
    return {"EB_DFT": EB_DFT, "EB_EXP": EB_EXP, "VR_DFT": VR_DFT, "VR_EXP": VR_EXP}


# ---------------------------------------------------------------------------
# Fractional-Mn descriptor interpolation and local-environment features
# ---------------------------------------------------------------------------
def eb_mn_frac(n_mn: float, EB: dict[str, float]) -> float:
    return (n_mn - 3) * EB["Mn4"] + (4 - n_mn) * EB["Mn3"]


def vr_mn_frac(n_mn: float, VR: dict[str, float]) -> float:
    return (n_mn - 3) * VR["Mn4_3"] + (4 - n_mn) * VR["Mn3_2"]


def n_mn_bulk(x_ce_bulk: float, x_mn_bulk: float) -> float:
    """Average Mn valence from bulk charge balance, n_Mn = 4 - X_Ce/X_Mn.

    When X_Mn = 0 there is no Mn, so the value is immaterial; return 4.0 to avoid a
    0 * inf in the (zero-weight) Mn term.
    """
    return 4.0 - x_ce_bulk / x_mn_bulk if x_mn_bulk > 0 else 4.0


def env_features(x_ce_loc: float, x_mn_loc: float, eb_mn: float, vr_mn: float,
                 EB: dict[str, float], VR: dict[str, float]) -> tuple[float, float]:
    """<Eb>, <Vr> for a local O environment: 4 A-site + 2 B-site neighbours (weights
    2/3 and 1/3). ``eb_mn``/``vr_mn`` are the (fractional-valence) Mn descriptors."""
    Eb = ((2.0 / 3.0) * (x_ce_loc * EB["Ce3"] + (1 - x_ce_loc) * EB["Ca2"])
          + (1.0 / 3.0) * (x_mn_loc * eb_mn + (1 - x_mn_loc) * EB["Ti4"]))
    Vr = ((2.0 / 3.0) * (x_ce_loc * VR["Ce3_0"] + (1 - x_ce_loc) * VR["Ca2_0"])
          + (1.0 / 3.0) * (x_mn_loc * vr_mn + (1 - x_mn_loc) * VR["Ti4_3"]))
    return Eb, Vr


def fit_coefficients(species: dict[str, dict[str, float]],
                     sites_file: Path = SITES_FILE) -> dict[str, dict[str, float]]:
    """Re-fit (a_Eb, b_Vr, intercept) by Huber regression on the 90 SCAN+U sites.

    The local environment of each site is (x_ce, x_mn) = (k_Ce/4, k_Mn/2); the Mn
    valence is set by the *bulk* composition (N_Ce/N_A, N_Mn/N_A). Returns coeffs for
    the SCAN+U ("DFT") and experimental ("EXP") descriptor parameterizations.
    """
    df = pd.read_csv(sites_file)
    df["XCe_bulk"] = df["NCe"] / df["NA"]
    df["XMn_bulk"] = df["NMn"] / df["NA"]
    df["n_Mn"] = df.apply(lambda r: n_mn_bulk(r["XCe_bulk"], r["XMn_bulk"]), axis=1)
    y = df["Ev"].to_numpy()

    coeffs: dict[str, dict[str, float]] = {}
    for tag, EB, VR in [("DFT", species["EB_DFT"], species["VR_DFT"]),
                        ("EXP", species["EB_EXP"], species["VR_EXP"])]:
        feats = df.apply(
            lambda r: env_features(r["xCe"], r["xMn"],
                                   eb_mn_frac(r["n_Mn"], EB), vr_mn_frac(r["n_Mn"], VR),
                                   EB, VR),
            axis=1, result_type="expand")
        m = HuberRegressor().fit(feats.to_numpy(), y)
        coeffs[tag] = {"a_Eb": float(m.coef_[0]), "b_Vr": float(m.coef_[1]),
                       "intercept": float(m.intercept_)}
    return coeffs


# ---------------------------------------------------------------------------
# Thermodynamics: oxygen chemical potential from gas reservoirs (NIST Shomate)
# ---------------------------------------------------------------------------
kB = 8.617333262145e-5            # eV / K
J_PER_MOL_PER_EV = 96485.33212

_O2_LOW = dict(A=31.32234, B=-20.23531, C=57.86644, D=-36.50624, E=-0.007374, F=-8.903471, G=246.7945, H=0.0)
_O2_MID = dict(A=30.03235, B=8.772972, C=-3.988133, D=0.788313, E=-0.741599, F=-11.32468, G=236.1663, H=0.0)
_O2_HIGH = dict(A=20.91111, B=10.72071, C=-2.020498, D=0.146449, E=9.245722, F=5.337651, G=237.6185, H=0.0)
_H2_LOW = dict(A=33.066178, B=-11.363417, C=11.432816, D=-2.772874, E=-0.158558, F=-9.980797, G=172.707974, H=0.0)
_H2_HIGH = dict(A=18.563083, B=12.257357, C=-2.859786, D=0.268238, E=1.977990, F=-1.147438, G=156.288133, H=0.0)
_H2O_MID = dict(A=30.09200, B=6.832514, C=6.793435, D=-2.534480, E=0.082139, F=-250.8810, G=223.3967, H=-241.8264)
_H2O_HIGH = dict(A=41.96426, B=8.622053, C=-1.499780, D=0.098119, E=-11.15764, F=-272.1797, G=219.7809, H=-241.8264)


def _shomate_G_eV(coeffs: dict[str, float], T: float) -> float:
    t = T / 1000.0
    H_kJ = (coeffs["A"] * t + coeffs["B"] * t**2 / 2 + coeffs["C"] * t**3 / 3
            + coeffs["D"] * t**4 / 4 - coeffs["E"] / t + coeffs["F"] - coeffs["H"])
    S_J = (coeffs["A"] * np.log(t) + coeffs["B"] * t + coeffs["C"] * t**2 / 2
           + coeffs["D"] * t**3 / 3 - coeffs["E"] / (2 * t**2) + coeffs["G"])
    H_J = 1000.0 * (H_kJ + coeffs["H"])
    return (H_J - T * S_J) / J_PER_MOL_PER_EV


def mu_O_from_O2(T: float, pO2_bar: float) -> float:
    c = _O2_LOW if T < 700 else (_O2_MID if T < 2000 else _O2_HIGH)
    return 0.5 * (_shomate_G_eV(c, T) + kB * T * np.log(pO2_bar))


def mu_O_from_H2O_H2(T: float, ratio_pH2_over_pH2O: float) -> float:
    g_h2o = _shomate_G_eV(_H2O_MID if T < 1700 else _H2O_HIGH, T)
    g_h2 = _shomate_G_eV(_H2_LOW if T < 1000 else _H2_HIGH, T)
    return g_h2o - g_h2 + kB * T * np.log(1.0 / ratio_pH2_over_pH2O)


# ---------------------------------------------------------------------------
# Binomial environment weighting and Delta-delta
# ---------------------------------------------------------------------------
def weights_local(x_ce_bulk: float, x_mn_bulk: float) -> list[tuple[float, float, float]]:
    """The 15 local environments (k_Ce/4, k_Mn/2) with binomial weights, normalised."""
    out, total = [], 0.0
    for k_ce in range(5):
        for k_mn in range(3):
            w = (comb(4, k_ce) * x_ce_bulk**k_ce * (1 - x_ce_bulk)**(4 - k_ce)
                 * comb(2, k_mn) * x_mn_bulk**k_mn * (1 - x_mn_bulk)**(2 - k_mn))
            out.append((k_ce / 4.0, k_mn / 2.0, w))
            total += w
    return [(xc, xm, w / total) for xc, xm, w in out]


def delta_red_ox(x_ce_bulk: float, x_mn_bulk: float, coef: dict[str, float],
                 EB: dict[str, float], VR: dict[str, float]) -> tuple[float, float]:
    """(delta_red, delta_ox) per formula unit at the cycling conditions."""
    n = n_mn_bulk(x_ce_bulk, x_mn_bulk)
    eb_mn, vr_mn = eb_mn_frac(n, EB), vr_mn_frac(n, VR)
    envs = weights_local(x_ce_bulk, x_mn_bulk)
    mu_red = mu_O_from_O2(T_RED, PO2_RED)
    mu_ox = mu_O_from_H2O_H2(T_OX, PH2_OVER_PH2O)

    def vac(mu: float, T: float) -> float:
        v = 0.0
        for x_ce_loc, x_mn_loc, w in envs:
            Eb, Vr = env_features(x_ce_loc, x_mn_loc, eb_mn, vr_mn, EB, VR)
            Ev = coef["a_Eb"] * Eb + coef["b_Vr"] * Vr + coef["intercept"]
            v += w / (1.0 + np.exp((Ev + mu) / (kB * T)))
        return v

    return 3.0 * vac(mu_red, T_RED), 3.0 * vac(mu_ox, T_OX)


def delta_delta(x_ce_bulk: float, x_mn_bulk: float, coef: dict[str, float],
                EB: dict[str, float], VR: dict[str, float]) -> float:
    dr, dox = delta_red_ox(x_ce_bulk, x_mn_bulk, coef, EB, VR)
    return dr - dox


def screen_grid(coef: dict[str, float], EB: dict[str, float], VR: dict[str, float],
                n: int = GRID_POINTS_PER_AXIS) -> pd.DataFrame:
    xs = np.linspace(0.0, 1.0, n)
    rows = [(x_ce, x_mn, delta_delta(x_ce, x_mn, coef, EB, VR))
            for x_ce in xs for x_mn in xs]
    return pd.DataFrame(rows, columns=["X_Ce", "X_Mn", "delta_delta"])


# The three synthesized compositions reported in main-text Table 3 (Series A,
# X_Mn = 2 X_Ce). Exact fractions so n_Mn = 3.5 reproduces cleanly.
TABLE3_COMPOSITIONS = [
    ("X_Ce = 0.25", 0.25, 0.50),
    ("X_Ce = 0.29", 21 / 72, 42 / 72),
    ("X_Ce = 0.33", 1 / 3, 2 / 3),
]


def main() -> None:
    species = load_species()
    coeffs = fit_coefficients(species)
    EB, VR = species["EB_DFT"], species["VR_DFT"]

    print("Species descriptors from", FEATURES_FILE.relative_to(ROOT))
    print("  Eb (DFT):", {k: round(v, 4) for k, v in EB.items()})
    print("  Vr (DFT):", {k: round(v, 4) for k, v in VR.items()})
    print("\nRe-fit CFM coefficients (Huber, fractional Mn, 2-feature):")
    for tag in ("DFT", "EXP"):
        c = coeffs[tag]
        print(f"  {tag}: a_Eb={c['a_Eb']:+.4f}  b_Vr={c['b_Vr']:+.4f}  intercept={c['intercept']:+.4f}")

    grid = screen_grid(coeffs["DFT"], EB, VR)
    grid.to_csv(GRID_CSV, index=False)
    print(f"\nWrote {GRID_CSV.relative_to(ROOT)}  ({len(grid)} points; "
          f"delta_delta in [{grid['delta_delta'].min():.4f}, {grid['delta_delta'].max():.4f}])")

    print("\nTable 3 CFM column (DFT):")
    for label, x_ce, x_mn in TABLE3_COMPOSITIONS:
        dd = delta_delta(x_ce, x_mn, coeffs["DFT"], EB, VR)
        print(f"  {label}: Delta-delta = {dd:.4f}  (rounds to {dd:.3f})")

    # Table 3 SCAN+U column: the same binomial environment weighting and gas
    # thermodynamics, but with the DFT site energies themselves (one Ev per
    # symmetry-distinct environment) in place of the CFM's linear model.
    ev_sites = pd.read_csv(HERE / "data" / "dft_ev_local_env.csv")
    mu_red = mu_O_from_O2(T_RED, PO2_RED)
    mu_ox = mu_O_from_H2O_H2(T_OX, PH2_OVER_PH2O)
    print("\nTable 3 SCAN+U column (binomial weights over the DFT site energies):")
    for (label, x_ce, x_mn), (m_lbl, n_lbl) in zip(
            TABLE3_COMPOSITIONS, [(0, 18), (0, 21), (0, 24)]):
        sub = ev_sites[(ev_sites["M"] == m_lbl) & (ev_sites["N"] == n_lbl)]
        e = sub["Ev"].to_numpy()
        k_ce = (sub["xCe"] * 4).round().astype(int)
        k_mn = (sub["xMn"] * 2).round().astype(int)
        w = np.array([comb(4, c) * x_ce**c * (1 - x_ce)**(4 - c)
                      * comb(2, k) * x_mn**k * (1 - x_mn)**(2 - k)
                      for c, k in zip(k_ce, k_mn)])
        w = w / w.sum()
        d_red = 3.0 * np.sum(w / (1.0 + np.exp((e + mu_red) / (kB * T_RED))))
        d_ox = 3.0 * np.sum(w / (1.0 + np.exp((e + mu_ox) / (kB * T_OX))))
        dd = d_red - d_ox
        print(f"  {label}: Delta-delta = {dd:.4f}  (rounds to {dd:.3f})")


if __name__ == "__main__":
    main()
