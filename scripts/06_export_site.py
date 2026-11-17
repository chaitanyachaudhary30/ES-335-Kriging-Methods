"""Collect result tables into docs/data/site.json for the website.

Only aggregates are exported (row counts, fold-level metrics, predicted
skills, kriging results, errors binned by distance, residuals averaged over
100 km cells with at least 5 test samples, density summaries, verdicts).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def records(path: Path) -> list[dict]:
    return clean(pd.read_csv(path)) if path.exists() else []


def clean(df: pd.DataFrame) -> list[dict]:
    """Records with NaN replaced by None, so the JSON is valid."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def export_results() -> None:
    """Phase 2: aggregates for sections 7 to 9 (docs/data/<dataset>_results.json). No per-sample values."""
    R = C.RESULTS
    for name in ("gemas", "lucas"):
        if not (R / "models" / f"{name}_model_metrics.csv").exists():
            continue
        met = pd.read_csv(R / "models" / f"{name}_model_metrics.csv")
        summ = pd.read_csv(R / "models" / f"{name}_model_summary.csv")
        pred = pd.read_csv(R / "hypotheses" / f"{name}_predictions.csv")
        psum = pd.read_csv(R / "hypotheses" / f"{name}_predictions_summary.csv")
        summ = summ.merge(psum[["split", "method", "pred_skill_mean", "pred_skill_sd", "pred_rank"]], on=["split", "method"])
        summ["obs_rank"] = summ.groupby("split")["skill_mean"].rank(ascending=False, method="min").astype(int)
        cov = (pd.read_csv(R / "models" / f"{name}_interval_coverage.csv")
               .groupby(["split", "method"], as_index=False)[["coverage_95", "mean_sd", "rmse_over_mean_sd"]].mean())
        tune = pd.read_csv(R / "tuning" / f"{name}_tuned_settings.csv")
        tune["settings"] = tune["settings"].map(json.loads)
        # Post-hoc check: predicted absolute OK RMSE (kriging variance) against the observed one, and the reference RMSE.
        st = pd.read_csv(R / "hypotheses" / f"{name}_training_stats.csv")
        chk = []
        for split in ("spatial", "random"):
            s = st[st.split == split].sort_values("fold")
            ok = pred[(pred.split == split) & (pred.method == "OK")].sort_values("fold")
            sill = (s.nugget + s.partial_sill).to_numpy()
            mm = met[met.split == split]
            chk.append({"split": split,
                        "sqrt_sill": float(np.sqrt(sill).mean()),
                        "mean_baseline_rmse": float(mm[mm.method == "Mean"].rmse.mean()),
                        "predicted_ok_rmse": float(np.sqrt(ok.pred_mse_ratio.to_numpy() * sill).mean()),
                        "observed_ok_rmse": float(mm[mm.method == "OK"].rmse.mean())})
        pd.DataFrame(chk).to_csv(R / "models" / f"{name}_apriori_absolute_check.csv", index=False)
        out = {
            "metrics": clean(met.round(6)),
            "summary": clean(summ.round(6)),
            "pred_per_fold": clean(pred.round(6)),
            "coverage": clean(cov.round(4)),
            "tuning": clean(tune.round(6)),
            "apriori_absolute_check": chk,
            "error_by_distance": records(R / "errors" / f"{name}_error_by_distance.csv"),
            "cell_residuals": records(R / "errors" / f"{name}_cell_residuals.csv"),
            "error_by_country": records(R / "errors" / f"{name}_error_by_country.csv"),
            "error_by_fold": records(R / "errors" / f"{name}_error_by_fold.csv"),
            "error_by_ph_range": records(R / "errors" / f"{name}_error_by_ph_range.csv"),
            "density": records(R / "density" / f"{name}_density_summary.csv"),
            "verdicts": records(R / "verdicts.csv"),
        }
        (C.DOCS_DATA / f"{name}_results.json").write_text(json.dumps(out, allow_nan=False, default=float))
        print("wrote", C.DOCS_DATA / f"{name}_results.json")


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
    export_results()


if __name__ == "__main__":
    main()
