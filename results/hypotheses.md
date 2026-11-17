# Pre-registered hypotheses

**Written by:** Chaitanya (Phase 1)
**Status:** fixed before any model is evaluated on test targets. This file is committed and merged before the baseline test evaluation (`scripts/05_baselines.py`) and before any Phase 2 model evaluation. Git history records the order.

All numbers below come from `scripts/04_hypotheses.py`, which reads **training pH values and test locations only**. Derivations are in `docs/theory.md`. LUCAS numbers will be added by the same script, in a separate commit, before any LUCAS test evaluation.

## The question these hypotheses test

Can statistics computed on the training data (sampling density, nugget ratio, range, trend strength, residual spatial structure) predict which method will have the lowest test RMSE in held-out regions?

## Training-fold statistics: GEMAS, spatial split

Ranges over the 5 outer folds (`results/hypotheses/gemas_training_stats.csv`):

| Statistic | Value |
|---|---|
| Training samples per fold | about 1,390 |
| Median distance, test point → nearest training point | 98–110 km (random split: 43–44 km) |
| Nugget ratio $\eta$ | 0.39–0.54 |
| Practical range | 345–673 km |
| Trend $R^2$, coordinates (in-sample) | 0.16–0.22 |
| Trend $R^2$, coordinates + covariates (in-sample) | 0.25–0.31 |
| Moran's I (8 nearest neighbours) | 0.44–0.49 |

## A-priori predicted skill (1 − RMSE / RMSE of the mean), GEMAS

From `results/hypotheses/gemas_predictions_summary.csv` (mean over folds; SD in brackets):

| Method | Spatial split | Random split |
|---|---|---|
| RK (RF trend) | 0.216 (0.030) | 0.227 (0.009) |
| RK (linear trend) | 0.203 (0.029) | 0.287 (0.024) |
| UK | 0.176 (0.022) | 0.273 (0.032) |
| Random Forest | 0.160 (0.034) | 0.232 (0.009) |
| LinReg (coords + covariates) | 0.120 (0.030) | 0.147 (0.006) |
| OK | 0.099 (0.022) | 0.211 (0.035) |
| GPR | 0.099 (0.022) (= OK by theory, Section 5) | 0.211 (0.035) |
| LinReg (coords) | 0.087 (0.019) | 0.101 (0.005) |
| Mean | 0 | 0 |

## Hypotheses

| ID | Hypothesis | Prediction for GEMAS | Basis |
|---|---|---|---|
| **H1** | When the coordinate trend extrapolates well between training blocks (inner-CV), UK beats OK on held-out regions. | UK skill > OK skill on the spatial split (predicted 0.18 vs 0.10). | Theory §4: the trend keeps working beyond the range, while OK falls back to the mean (Result 1). |
| **H2** | The nugget caps what coordinate-only kriging can achieve: observed RMSE$_{OK}$/RMSE$_{mean}$ ≥ $\sqrt{\eta}$ of the training fold. | OK and GPR RMSE ratio ≥ 0.62–0.73 per fold, i.e. skill ≤ 0.27–0.38. Predicted skill is far below the cap (0.10) because test points are 100 km away. | Theory §3, Result 2. |
| **H3** | When covariates add explanatory power and the residuals stay spatially correlated, RK is the best method. | An RK variant has the highest skill on the spatial split; RK (linear) beats UK. | Theory §4; residual nugget ratio ≈ 0.45 < 1. |
| **H4** | The gain of kriging over its own trend shrinks with distance from the training data and is about zero beyond the range. | For test points with nearest training point > 250 km, RK skill − trend-only skill ≤ 0.02. In the LUCAS sparsity experiment, this gain falls monotonically as the training fraction falls. | Theory §3, Result 1 and the table of $r(d)$. |
| **H5** | Denser data gives a larger kriging gain on the same held-out regions. | OK skill on LUCAS > OK skill on GEMAS, using the same block-to-fold mapping. | Theory §3: shorter distances → larger $r(d)$. |
| **H6** | A random split overstates spatial generalisation. | For OK, GPR, UK and RK, random-split skill > spatial-split skill. Predicted gap for OK ≈ 0.11. | Theory §6. |
| **H7** (main question) | The a-priori predictions rank the methods correctly. | Spearman correlation between predicted and observed skill across the 9 methods ≥ 0.7 on the spatial split, and the predicted best method is within one fold-SD of the observed best. | Combines H1–H5. |
| **H8** | GPR and OK give similar accuracy. | \|skill(GPR) − skill(OK)\| ≤ 0.03 on both splits. | Theory §5. |

## Decision rule for each hypothesis

For a comparison "A beats B", compute the per-fold difference skill(A) − skill(B) on the 5 outer folds.

- **Supported:** the mean difference has the predicted sign **and** at least 4 of the 5 folds agree.
- **Not supported:** the mean difference has the opposite sign **and** at least 4 of the 5 folds agree.
- **Inconclusive:** anything else. Differences within fold-to-fold variation are not counted as evidence.

Threshold hypotheses (H2, H4, H8) are judged on the per-fold values against the stated threshold, with the same 4-of-5 rule.

## What would count as an answer

- If H7 is supported, training statistics are a useful guide for choosing a method for unseen regions in these data.
- If H7 fails, the failures of the individual hypotheses show which assumption broke (stationarity, trend transfer, nugget estimation), and that is reported as the finding.
