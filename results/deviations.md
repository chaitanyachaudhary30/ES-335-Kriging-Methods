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
