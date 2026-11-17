"""Step 2.6: apply the pre-registered decision rule (results/hypotheses.md, with the multi-part rules of
results/deviations.md D2) to the observed results.

Reads (aggregates and metrics only)
  results/models/gemas_model_metrics.csv, results/hypotheses/gemas_predictions.csv,
  results/errors/gemas_error_by_distance_fold.csv, results/errors/gemas_error_by_distance.csv,
  results/density/gemas_density_metrics.csv
Writes
  results/verdicts.csv   one row per hypothesis part and one "overall" row per hypothesis
  results/verdicts.md    the same as a readable table
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.models import METHODS  # noqa: E402

R = C.RESULTS
SUP, NOT, INC, NT = "supported", "not supported", "inconclusive", "not testable"
RK = ["RK (linear)", "RK (RF)"]
F = range(C.N_FOLDS)


def rule_diff(d):
    """A beats B: d = per-fold skill(A) - skill(B)."""
    d = np.asarray(d, float)
    if (d > 0).sum() >= 4 and d.mean() > 0:
        return SUP
    if (d < 0).sum() >= 4 and d.mean() < 0:
        return NOT
    return INC


def rule_pass(ok):
    """Threshold hypotheses: ok = per-fold booleans."""
    ok = np.asarray(ok, bool)
    if ok.sum() >= 4:
        return SUP
    if (~ok).sum() >= 4:
        return NOT
    return INC


def combine(verdicts):
    if all(v == SUP for v in verdicts):
        return SUP
    if all(v == NOT for v in verdicts):
        return NOT
    return INC


def nums(a, d=3):
    return "[" + ", ".join(f"{x:.{d}f}" for x in a) + "]"


def main() -> None:
    m = pd.read_csv(R / "models" / "gemas_model_metrics.csv")
    pr = pd.read_csv(R / "hypotheses" / "gemas_predictions.csv")
    skill = lambda split, method: m[(m.split == split) & (m.method == method)].sort_values("fold")["skill"].to_numpy()  # noqa: E731
    rows = []

    def add(h, part, statement, evidence, verdict, key=""):
        rows.append({"hypothesis": h, "part": part, "statement": statement, "evidence": evidence, "key_number": key, "verdict": verdict})

    def overall(h, statement, key):
        v = [r["verdict"] for r in rows if r["hypothesis"] == h]
        add(h, "overall", statement, "", combine(v), key)

    # H1
    d = skill("spatial", "UK") - skill("spatial", "OK")
    add("H1", "UK - OK, spatial", "UK skill > OK skill", f"per-fold differences {nums(d)}; mean {d.mean():+.3f}; UK > OK in {(d > 0).sum()}/5 folds", rule_diff(d), f"{d.mean():+.3f}")
    overall("H1", "UK beats OK on held-out regions", f"UK - OK = {d.mean():+.3f}")

    # H2
    eta = pr[(pr.split == "spatial") & (pr.method == "OK")].sort_values("fold")["sqrt_eta_bound"].to_numpy()
    for meth in ("OK", "GPR"):
        sk = skill("spatial", meth)
        ok = (1 - sk) >= eta
        add("H2", f"{meth}, spatial", "RMSE ratio >= sqrt(eta)", f"RMSE ratios {nums(1 - sk)} vs sqrt(eta) {nums(eta)}; bound holds in {ok.sum()}/5 folds", rule_pass(ok), f"ratio {np.mean(1 - sk):.3f} vs {eta.mean():.3f}")
    overall("H2", "Nugget bound holds for OK and GPR", f"bound holds; OK RMSE ratio {np.mean(1 - skill('spatial', 'OK')):.3f} >= {eta.mean():.3f}")

    # H3
    sp = m[m.split == "spatial"].pivot(index="fold", columns="method", values="skill")
    best = sp.idxmax(axis=1)
    n_rk = int(best.isin(RK).sum())
    n_other = int((~best.isin(RK)).sum())
    va = SUP if n_rk >= 4 else NOT if n_other >= 4 else INC
    add("H3", "(a) best method is an RK variant", "an RK variant has the highest skill", f"best method per fold: {list(best)}; mean skills: " + ", ".join(f"{k} {v:.3f}" for k, v in sp.mean().sort_values(ascending=False).items()), va, f"best mean: {sp.mean().idxmax()} {sp.mean().max():.3f}")
    d = skill("spatial", "RK (linear)") - skill("spatial", "UK")
    add("H3", "(b) RK (linear) - UK", "RK (linear) beats UK", f"per-fold differences {nums(d)}; mean {d.mean():+.3f}", rule_diff(d), f"{d.mean():+.3f}")
    overall("H3", "RK is the best method and RK (linear) beats UK", f"best: {sp.mean().idxmax()}; RK(lin) - UK = {d.mean():+.3f}")

    # H4 (a): points more than 250 km from the training data, spatial split
    bd = pd.read_csv(R / "errors" / "gemas_error_by_distance_fold.csv")
    bd = bd[(bd.split == "spatial") & (bd["bin"] == ">250")]
    gains, ns = [], []
    for f in F:
        g = bd[bd.fold == f].set_index("method")
        if len(g) == 0 or g.loc["Mean", "n"] < 15:
            continue
        sk = lambda meth: 1 - g.loc[meth, "rmse"] / g.loc["Mean", "rmse"]  # noqa: E731
        gains.append(sk("RK (linear)") - sk("LinReg (coords + covariates)"))
        ns.append(int(g.loc["Mean", "n"]))
    pooled = pd.read_csv(R / "errors" / "gemas_error_by_distance.csv")
    pooled = pooled[(pooled.split == "spatial") & (pooled["bin"] == ">250")].set_index("method")
    psk = lambda meth: 1 - pooled.loc[meth, "rmse"] / pooled.loc["Mean", "rmse"]  # noqa: E731
    pooled_txt = (f"pooled over spatial folds (n={int(pooled.loc['Mean', 'n'])}): RK (linear) {psk('RK (linear)'):.3f} - LinReg (coords + covariates) {psk('LinReg (coords + covariates)'):.3f} = {psk('RK (linear)') - psk('LinReg (coords + covariates)'):+.3f}; "
                  f"RK (RF) {psk('RK (RF)'):.3f} - Random Forest {psk('Random Forest'):.3f} = {psk('RK (RF)') - psk('Random Forest'):+.3f}")
    if len(gains) < 4:
        va = INC
        ev = f"only {len(gains)} folds have at least 15 test points beyond 250 km (n = {ns}); gains {nums(gains)}; {pooled_txt}"
    else:
        va = rule_pass(np.array(gains) <= 0.02)
        ev = f"folds with >= 15 points: {len(gains)}, n = {ns}; gains {nums(gains)}; {pooled_txt}"
    add("H4", "(a) gain beyond 250 km", "RK (linear) skill - trend-only skill <= 0.02 for points > 250 km", ev, va, f"pooled gain {psk('RK (linear)') - psk('LinReg (coords + covariates)'):+.3f}")

    # H4 (b): density experiment
    dm = pd.read_csv(R / "density" / "gemas_density_metrics.csv")
    pf = dm.groupby(["fold", "fraction", "method"], as_index=False)["skill"].mean()
    piv = pf.pivot(index=["fold", "fraction"], columns="method", values="skill")
    gain = (piv["RK (linear)"] - piv["LinReg (coords + covariates)"]).unstack("fraction")[[1.0, 0.5, 0.25, 0.1]]
    mono = np.array([(np.diff(gain.loc[f].to_numpy()) <= 1e-12).all() for f in gain.index])
    add("H4", "(b) gain vs density", "kriging gain falls as the training fraction falls", f"mean gain at fractions 1.0/0.5/0.25/0.1: {nums(gain.mean().to_numpy())}; folds where the gain never rises as data are removed: {mono.sum()}/5", rule_pass(mono), f"gain {gain.mean()[1.0]:+.3f} -> {gain.mean()[0.1]:+.3f}")
    overall("H4", "Kriging gain vanishes far from data and falls with density", "see parts")

    # H5
    add("H5", "overall", "Denser data gives a larger kriging gain (LUCAS vs GEMAS)", "LUCAS 2018 access was not granted (deviation D1)", NT, "not testable")

    # H6
    h6 = []
    for meth in ("OK", "GPR", "UK", "RK (linear)", "RK (RF)"):
        d = skill("random", meth) - skill("spatial", meth)
        v = rule_diff(d)
        h6.append(v)
        add("H6", meth, "random-split skill > spatial-split skill", f"per-fold differences {nums(d)}; mean {d.mean():+.3f}", v, f"{d.mean():+.3f}")
    overall("H6", "A random split overstates skill", f"OK: {skill('random', 'OK').mean():.3f} (random) vs {skill('spatial', 'OK').mean():.3f} (spatial)")

    # H7
    pp = pr[pr.split == "spatial"].pivot(index="fold", columns="method", values="pred_skill")[METHODS]
    obs = sp[METHODS]
    rho = np.array([spearmanr(pp.loc[f], obs.loc[f]).statistic for f in F])
    rho_mean = spearmanr(pp.mean(), obs.mean()).statistic
    add("H7", "(a) Spearman rho", "rho >= 0.7 in at least 4 of 5 folds", f"per-fold rho {nums(rho, 2)}; rho of fold means {rho_mean:.2f}", rule_pass(rho >= 0.7), f"rho (fold means) {rho_mean:.2f}")
    pbest, obest = pp.mean().idxmax(), obs.mean().idxmax()
    sd_best = obs[obest].std()
    gap = obs.mean()[obest] - obs.mean()[pbest]
    add("H7", "(b) best method", "predicted best is within one fold SD of the observed best", f"predicted best {pbest} (observed {obs.mean()[pbest]:.3f}); observed best {obest} ({obs.mean()[obest]:.3f}, fold SD {sd_best:.3f}); gap {gap:.3f}", SUP if gap <= sd_best else NOT, f"gap {gap:.3f} vs SD {sd_best:.3f}")
    overall("H7", "Predicted ranking matches the observed ranking", f"rho (fold means) = {rho_mean:.2f}")

    # H8
    for split in ("spatial", "random"):
        d = np.abs(skill(split, "GPR") - skill(split, "OK"))
        add("H8", split, "abs(skill(GPR) - skill(OK)) <= 0.03", f"per-fold absolute differences {nums(d)}; <= 0.03 in {(d <= 0.03).sum()}/5 folds", rule_pass(d <= 0.03), f"max {d.max():.3f}")
    overall("H8", "GPR and OK are equivalent", f"mean absolute difference, spatial {np.abs(skill('spatial', 'GPR') - skill('spatial', 'OK')).mean():.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(R / "verdicts.csv", index=False)

    lines = ["# Hypothesis verdicts", "",
             "Produced by `scripts/08_verdicts.py`. The decision rule is the one in `results/hypotheses.md` (mean difference with the predicted sign and at least 4 of 5 folds agree), "
             "with the multi-part rules of `results/deviations.md` D2. Skill = 1 - RMSE / RMSE of the mean baseline of the same fold. The rules and thresholds were fixed in D2 before any Phase 2 model was scored on test data and were not changed afterwards. Fold *i* of the random split is paired with fold *i* of the spatial split in H6, although their test sets differ.", "",
             "| Hypothesis | Verdict | Key number |", "|---|---|---|"]
    for r in out[out.part == "overall"].itertuples():
        lines.append(f"| {r.hypothesis}: {r.statement} | **{r.verdict}** | {r.key_number} |")
    lines += ["", "## Evidence by part", "", "| Hypothesis | Part | Statement | Evidence | Verdict |", "|---|---|---|---|---|"]
    for r in out[out.part != "overall"].itertuples():
        lines.append(f"| {r.hypothesis} | {r.part} | {r.statement} | {r.evidence} | {r.verdict} |")
    (R / "verdicts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
