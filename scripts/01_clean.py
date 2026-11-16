"""Step 1.3: clean both surveys, attach covariates, and write the row-count table.

Usage:  python -m scripts.01_clean   (or)   python scripts/01_clean.py
LUCAS is skipped with a message if its CSV has not been downloaded yet.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import load_gemas, load_lucas  # noqa: E402


def main() -> None:
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "data_quality").mkdir(parents=True, exist_ok=True)
    logs = []

    gemas, log = load_gemas()
    gemas.to_csv(C.DATA_PROCESSED / "gemas_clean.csv", index=False)
    logs.append(log)
    print(f"GEMAS: {len(gemas)} rows")

    try:
        lucas, log = load_lucas()
        lucas.to_csv(C.DATA_PROCESSED / "lucas_clean.csv", index=False)
        logs.append(log)
        print(f"LUCAS: {len(lucas)} rows")
    except FileNotFoundError as err:
        print(f"LUCAS skipped: {err}")

    table = pd.concat(logs, ignore_index=True)
    table.to_csv(C.RESULTS / "data_quality" / "row_counts.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
