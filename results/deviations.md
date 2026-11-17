# Deviations from the proposal and the pre-registered plan

`results/hypotheses.md` is left unchanged so that the pre-registration stays as committed. Every change to the plan is recorded here instead, with its reason and its effect on the conclusions.

## D1. LUCAS 2018 Topsoil could not be used

- **What happened:** we submitted the ESDAC registration form for LUCAS 2018 Topsoil (https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data) but did not receive access to the data before the submission deadline.
- **Decision:** all results use GEMAS only. The LUCAS loading code (`src/data.py: load_lucas`) stays in the repository, and every script skips LUCAS when its file is missing, so the analysis can be extended if access is granted later. Its LUCAS column handling has not been tested on the real file.
- **Effect on the hypotheses:**
  - **H5** (denser LUCAS data gives a larger kriging gain than GEMAS) cannot be tested. It is reported as *not testable*, not as supported or rejected.
  - **H4**, second part (gain falls as LUCAS training density falls): the density experiment is run by thinning the **GEMAS** training folds instead (fractions 100%, 50%, 25%, 10%, 5 seeds each, test folds unchanged, same nested-subset rule as `src/splits.py`). This is a deviation: GEMAS can only be made *sparser* than it already is, so the experiment covers spacings from about 42 km upwards, and says nothing about denser sampling.
  - **H1–H3, H6–H8** are unaffected: they were stated for GEMAS.
- **Effect on the claim:** the study now covers one survey. Conclusions about how sampling density changes the best method rest on the thinning experiment, within one sampling design and one lab, instead of on a comparison of two surveys.

## D2. How the decision rule is applied to hypotheses with several parts

Written and committed in the `saksham/models` pull request, before any Phase 2 model was scored on test data. `hypotheses.md` gives one decision rule for "A beats B" comparisons. Several hypotheses contain more than one comparison, so the following mechanical rules are fixed here.

- **Unit:** a comparison is judged on the 5 outer folds of the stated split. Fold *i* of the random split is paired with fold *i* of the spatial split (H6 only).
- **Several parts or methods:** the hypothesis is *supported* if every part is supported, *not supported* if every part is not supported, and *inconclusive* otherwise (mixed results count as inconclusive).
- **H2:** one part each for OK and GPR; a fold passes if observed RMSE / RMSE(mean) is at least the square root of that fold's training nugget ratio.
- **H3:** part (a) an RK variant has the highest skill in at least 4 of 5 folds on the spatial split (not supported if another method is best in at least 4 folds); part (b) RK (linear) beats UK by the standard rule.
- **H4:** part (a) for test points more than 250 km from the nearest training point, skill(RK (linear)) minus skill(LinReg coords + covariates) is at most 0.02. Only folds with at least 15 such points count; if fewer than 4 folds qualify, part (a) is inconclusive and the pooled value over all spatial folds is reported as extra information. RK (RF) against Random Forest is reported as a secondary check. Part (b) in the density experiment: the same kriging gain, averaged over the 5 thinning seeds, does not increase as the training fraction decreases (1.0, 0.5, 0.25, 0.1), in at least 4 of 5 folds.
- **H6:** one part each for OK, GPR, UK, RK (linear), RK (RF): random-split skill > spatial-split skill.
- **H7:** part (a) the Spearman correlation between predicted and observed skill over the 9 methods is at least 0.7 in at least 4 of 5 spatial folds (the correlation of the fold means is reported too); part (b) the observed skill of the method predicted to be best (fold means) is within one fold SD of the observed skill of the observed best method.
- **H8:** one part per split; a fold passes if the absolute difference of GPR and OK skill is at most 0.03.
- **H5:** not testable (D1).

**Effect:** none on the data or models. These rules only fix how the pre-registered text is turned into code.

## D3. What the commit history can and cannot show about timing

Added after testing, as a clarification. Nothing in the analysis changed.

- **What the history shows.** `results/hypotheses.md` was committed in `45c3ebe` and has not changed since (`git diff 45c3ebe main -- results/hypotheses.md` is empty). GitHub's push log (repository activity API) records the branch holding it on GitHub at 2026-11-16 17:18 UTC, and the first test scores (`chaitanya/baselines`, commit `36c61d9`) pushed at 18:03 UTC. D2 was pushed in `saksham/models` at 2026-11-17 14:22 UTC, before the tuning (14:32 UTC) and evaluation (14:33 UTC) branches.
- **What it cannot show.** A commit records when files were saved, not when a script was run. Commit times alone therefore cannot prove that no test score had been looked at before a hypothesis or rule was written. For Phase 1, `05_baselines.py` runs in about 15 seconds, so the 62 seconds between the hypotheses commit and the baselines commit is consistent with the stated order. For Phase 2, the evaluation commit came 83 seconds after the tuning commit, while `07_models.py` takes a few minutes, so the evaluation must have been run before the tuning commit was made. The history shows that D2 was committed before the evaluation results, not that it was written before the evaluation was run.
- **What does not depend on timing.** The a-priori predictions are produced by `scripts/04_hypotheses.py`, which reads training pH values and test *locations* only (see the per-fold loop, which builds the training frame from rows outside the test fold). Anyone can rerun it and get the committed predictions without touching test pH. The verdicts are produced mechanically by `scripts/08_verdicts.py` from the rules in `hypotheses.md` and D2.

