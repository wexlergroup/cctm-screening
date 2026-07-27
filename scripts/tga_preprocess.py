"""Extract the committed TGA figure data from the raw instrument workbooks.

The raw thermogravimetric analysis exports (one workbook per run, ASU; kept
verbatim under ``figures/data/raw/tga/``) carry the buoyancy-corrected mass
signal and the derived off-stoichiometry delta on sheet
``Sample-blank TGA ddd``. This script extracts the three columns the ESI TGA
figure consumes and writes one CSV per run to ``figures/data/``:

    time_h        -- ``Time/min`` / 60
    temperature_C -- ``##Temp./°C``
    delta         -- ``delta`` (off-stoichiometry referenced to the sample's
                     starting oxygen content)

Values are written in ``%.5g`` format, which reproduces the committed
``tga_*.csv`` files byte for byte (verified against all six runs).

Protocol A: 1350 °C (0.5 h) -> 1300 °C (1 h) reduction in Ar;
protocol B: 1350 °C (1 h) -> 1300 °C (0.5 h) reduction in Ar;
both reoxidize at 900 °C (0.5 h) in 20% O2/Ar.

Usage (paths are anchored to this file):

    python tga_preprocess.py
"""
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "figures" / "data" / "raw" / "tga"
DATA_DIR = SCRIPT_DIR.parent / "figures" / "data"

SHEET = "Sample-blank TGA ddd"

# (raw workbook, output CSV)
RUNS = [
    ("20240821 CCTM 2112 R1350+1300  1h [240503b].xlsx",
     "tga_CCTM2112_protocol_A.csv"),
    ("20241031 CCTM 2112 R1350 h+1300 30 min (240503b).xlsx",
     "tga_CCTM2112_protocol_B.csv"),
    ("20240830 CCTM 3122 R1350+1300 [240503b].xlsx",
     "tga_CCTM3122_protocol_A.csv"),
    ("20240918 CCTM 3122 R1350 h +1300 30min(240503b).xlsx",
     "tga_CCTM3122_protocol_B.csv"),
    ("20240905 CCTM1771014 R1350+1300 1h[240620b].xlsx",
     "tga_CCTM1771014_protocol_A.csv"),
    ("20240924 CCTM1771014 R1350 h+1300 30min (240620b).xlsx",
     "tga_CCTM1771014_protocol_B.csv"),
]


def main() -> None:
    for workbook, csv_name in RUNS:
        raw = pd.read_excel(RAW_DIR / workbook, sheet_name=SHEET)
        out = pd.DataFrame({
            "time_h": raw["Time/min"] / 60.0,
            "temperature_C": raw["##Temp./°C"],
            "delta": raw["delta"],
        })
        out_path = DATA_DIR / csv_name
        out.to_csv(out_path, index=False, float_format="%.5g")
        print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
