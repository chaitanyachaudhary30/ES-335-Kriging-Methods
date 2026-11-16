"""Collect result tables into docs/data/site.json for the website.

Only aggregates are exported (row counts, fold-level metrics, predicted
skills). Phase 2 extends this script with the kriging results.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def main() -> None:
    site = {
        "row_counts": records(C.RESULTS / "data_quality" / "row_counts.csv"),
        "split_summary": records(C.RESULTS / "splits" / "split_summary.csv"),
        "datasets": {},
        "config": {"block_km": C.BLOCK_KM, "n_folds": C.N_FOLDS, "seed": C.SEED},
    }
    for name in ("gemas", "lucas"):
        b = C.RESULTS / "baselines" / f"{name}_baseline_metrics.csv"
        p = C.RESULTS / "hypotheses" / f"{name}_predictions_summary.csv"
        if not p.exists():
            continue
        site["datasets"][name] = {
            "baseline_metrics": records(b),
            "baseline_summary": records(C.RESULTS / "baselines" / f"{name}_baseline_summary.csv"),
            "predictions_summary": records(p),
        }
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (C.DOCS_DATA / "site.json").write_text(json.dumps(site))
    print("wrote", C.DOCS_DATA / "site.json", "datasets:", list(site["datasets"]))


if __name__ == "__main__":
    main()
