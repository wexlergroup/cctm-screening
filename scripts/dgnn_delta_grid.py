"""dGNN screening: redox-capacity (Delta-delta) grid from the fine-tuned dGNN.

Converts the fine-tuned defect-GNN predictions for the sampled oxygen
environments of each bulk composition
(``figures/data/dgnn_screen_finetuned.csv``) into the (X_Ce, X_Mn)
redox-capacity grid ``figures/data/dgnn_delta_grid.csv`` that
``figures/figure5.py`` renders as Fig. 5b.

For each composition, the predicted formation energies are mapped to vacancy
occupancies by Fermi-Dirac statistics at the oxygen chemical potential of each
half-cycle (NIST Shomate gas thermochemistry, identical to
``figures/cfm_model.py``), and delta = 3 * <occupancy>:

    delta_red at 1350 C, pO2 = 1e-4 bar
    delta_ox  at  850 C, pH2 / pH2O = 1e-3
    Delta-delta = delta_red - delta_ox

This is a self-contained copy of the script used for the published screening;
only the I/O plumbing has been adapted to read the committed data in this
repository. Run from anywhere (paths are anchored to this file):

    python dgnn_delta_grid.py

Writes ``figures/data/dgnn_delta_grid.csv`` (consumed by figure5.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Physical constants
kB_EV_PER_K = 8.617333262145e-5
J_PER_MOL_PER_EV = 96485.33212

# Gas phase parameters
T_RED = 1623.15  # K
PO2_RED = 1e-4  # bar
T_OX = 1123.15  # K
RATIO_WS = 1.0 / 1000.0  # PH2 / PH2O

# File locations
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "figures" / "data" / "dgnn_screen_finetuned.csv"
OUTPUT_FILE = ROOT / "figures" / "data" / "dgnn_delta_grid.csv"


# Shomate coefficients (NIST) include the user's enthalpy offset
O2_SHOMATE_LOW = {
    "A": 31.32234,
    "B": -20.23531,
    "C": 57.86644,
    "D": -36.50624,
    "E": -0.007374,
    "F": -8.903471,
    "G": 246.7945,
    "H": 0.0,
}
O2_SHOMATE_MID = {
    "A": 30.03235,
    "B": 8.772972,
    "C": -3.988133,
    "D": 0.788313,
    "E": -0.741599,
    "F": -11.32468,
    "G": 236.1663,
    "H": 0.0,
}
O2_SHOMATE_HIGH = {
    "A": 20.91111,
    "B": 10.72071,
    "C": -2.020498,
    "D": 0.146449,
    "E": 9.245722,
    "F": 5.337651,
    "G": 237.6185,
    "H": 0.0,
}
H2_SHOMATE_LOW = {
    "A": 33.066178,
    "B": -11.363417,
    "C": 11.432816,
    "D": -2.772874,
    "E": -0.158558,
    "F": -9.980797,
    "G": 172.707974,
    "H": 0.0,
}
H2_SHOMATE_HIGH = {
    "A": 18.563083,
    "B": 12.257357,
    "C": -2.859786,
    "D": 0.268238,
    "E": 1.97799,
    "F": -1.147438,
    "G": 156.288133,
    "H": 0.0,
}
H2O_SHOMATE_MID = {
    "A": 30.09200,
    "B": 6.832514,
    "C": 6.793435,
    "D": -2.53448,
    "E": 0.082139,
    "F": -250.8810,
    "G": 223.3967,
    "H": -241.8264,
}
H2O_SHOMATE_HIGH = {
    "A": 41.96426,
    "B": 8.622053,
    "C": -1.49978,
    "D": 0.098119,
    "E": -11.15764,
    "F": -272.1797,
    "G": 219.7809,
    "H": -241.8264,
}


@dataclass(frozen=True)
class DeltaResult:
    X_Ce: float
    X_Mn: float
    delta_red: float
    delta_ox: float
    delta_delta: float
    vacancy_pct_red: float
    vacancy_pct_ox: float


def _shomate_enthalpy_entropy(coeffs: dict[str, float], temperature: float) -> tuple[float, float]:
    t = temperature / 1000.0
    a, b, c, d, e, f, g, h0 = (
        coeffs["A"],
        coeffs["B"],
        coeffs["C"],
        coeffs["D"],
        coeffs["E"],
        coeffs["F"],
        coeffs["G"],
        coeffs["H"],
    )
    h_kj_per_mol = a * t + b * t**2 / 2.0 + c * t**3 / 3.0 + d * t**4 / 4.0 - e / t + f - h0
    s_j_per_mol_k = a * np.log(t) + b * t + c * t**2 / 2.0 + d * t**3 / 3.0 - e / (2.0 * t**2) + g
    h_j_per_mol = 1000.0 * (h_kj_per_mol + h0)
    return h_j_per_mol, s_j_per_mol_k


def g_o2_std(temperature: float) -> float:
    if temperature < 700.0:
        coeffs = O2_SHOMATE_LOW
    elif temperature < 2000.0:
        coeffs = O2_SHOMATE_MID
    else:
        coeffs = O2_SHOMATE_HIGH
    enthalpy, entropy = _shomate_enthalpy_entropy(coeffs, temperature)
    return enthalpy - temperature * entropy


def g_h2_std(temperature: float) -> float:
    coeffs = H2_SHOMATE_LOW if temperature < 1000.0 else H2_SHOMATE_HIGH
    enthalpy, entropy = _shomate_enthalpy_entropy(coeffs, temperature)
    return enthalpy - temperature * entropy


def g_h2o_std(temperature: float) -> float:
    coeffs = H2O_SHOMATE_MID if temperature < 1700.0 else H2O_SHOMATE_HIGH
    enthalpy, entropy = _shomate_enthalpy_entropy(coeffs, temperature)
    return enthalpy - temperature * entropy


def mu_o_from_o2(temperature: float, p_o2_bar: float = 1.0) -> float:
    g_o2 = g_o2_std(temperature) / J_PER_MOL_PER_EV
    return 0.5 * (g_o2 + kB_EV_PER_K * temperature * np.log(p_o2_bar))


def mu_o_from_h2o_h2(temperature: float, p_h2_over_p_h2o: float) -> float:
    delta_g = (g_h2o_std(temperature) - g_h2_std(temperature)) / J_PER_MOL_PER_EV
    return delta_g + kB_EV_PER_K * temperature * np.log(1.0 / p_h2_over_p_h2o)


def vacancy_fraction(formation_energies: np.ndarray, mu_o: float, temperature: float) -> float:
    beta = 1.0 / (kB_EV_PER_K * temperature)
    occupations = 1.0 / (1.0 + np.exp(beta * (formation_energies + mu_o)))
    return float(np.mean(occupations))


def compute_delta_values(data: pd.DataFrame) -> list[DeltaResult]:
    mu_red = mu_o_from_o2(T_RED, PO2_RED)
    mu_ox = mu_o_from_h2o_h2(T_OX, RATIO_WS)

    results: list[DeltaResult] = []
    for (xce, xmn), group in data.groupby(["X_Ce", "X_Mn"]):
        energies = group["avgDefCv"].to_numpy(dtype=float)
        vacancy_red = vacancy_fraction(energies, mu_red, T_RED)
        vacancy_ox = vacancy_fraction(energies, mu_ox, T_OX)
        delta_red = 3.0 * vacancy_red
        delta_ox = 3.0 * vacancy_ox
        results.append(
            DeltaResult(
                X_Ce=float(xce),
                X_Mn=float(xmn),
                delta_red=delta_red,
                delta_ox=delta_ox,
                delta_delta=delta_red - delta_ox,
                vacancy_pct_red=100.0 * vacancy_red,
                vacancy_pct_ox=100.0 * vacancy_ox,
            )
        )
    return results


def results_to_dataframe(results: list[DeltaResult]) -> pd.DataFrame:
    frame = pd.DataFrame([result.__dict__ for result in results])
    return frame.sort_values(by=["X_Ce", "X_Mn"]).reset_index(drop=True)


def main() -> None:
    dataset = pd.read_csv(DATA_FILE)
    results = compute_delta_values(dataset)
    table = results_to_dataframe(results)
    table[["X_Ce", "X_Mn", "delta_delta"]].to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE.relative_to(ROOT)}  ({len(table)} compositions; "
          f"delta_delta in [{table['delta_delta'].min():.4f}, {table['delta_delta'].max():.4f}])")


if __name__ == "__main__":
    main()
