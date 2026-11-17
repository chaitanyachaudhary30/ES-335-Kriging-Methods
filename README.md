# Can a dataset's statistics predict the best kriging method? Soil pH in unseen European regions

ES 335 course project by **Chaitanya** and **Saksham**.

- **Website:** https://chaitanyachaudhary30.github.io/ES-335-Kriging-Methods/
- **Question:** Can statistics of the training data (sample size, spacing, nugget ratio, range, trend strength) predict which kriging-based method gives the lowest error when predicting topsoil pH (CaCl$_2$) in geographically held-out regions?
- **Unit of analysis:** one row = one georeferenced agricultural topsoil sample (0–20 cm) with its measured pH.

> Status: complete. Phase 1 (data, statistics, splits, theory, hypotheses, baselines, website skeleton) and Phase 2 (models, tuning, evaluation, density experiment, error analysis, verdicts, results pages) are merged on `main`. The `phase1-handoff` tag marks the end of Phase 1.
>
> **LUCAS 2018 was not used.** We submitted the ESDAC registration form but did not receive access before the deadline, so all results use GEMAS only. See `results/deviations.md` for the effect on the hypotheses.

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | Paths, projection (EPSG:3035), seed, block size, country list, covariates |
| `src/data.py` | Loading, filtering and quality checks for GEMAS and LUCAS; covariate extraction |
| `src/features.py` | Allowed prediction-time inputs (coordinates + WorldClim covariates) |
| `src/spatial_stats.py` | Nearest-neighbour distances, Moran's I, trend $R^2$, variograms |
| `src/splits.py` | Spatial block folds, random folds, LUCAS sparsity subsets |
| `src/kriging_theory.py` | Kriging variance used for the a-priori error predictions |
| `src/baselines.py`, `src/metrics.py` | Mean / Linear Regression / Random Forest baselines; RMSE, MAE, skill |
| `src/models.py`, `src/tuning.py`, `src/apriori.py` | OK, UK, GPR, RK with one interface; inner-CV tuning; a-priori skill on any training set |
| `scripts/01_clean.py` to `08_verdicts.py` | Pipeline steps, run in order (see Run order) |
| `tests/` | `test_models.py` (synthetic data), `check_site.py` (headless-browser check of the website) |
| `data/splits/` | Exact fold assignment for every sample (IDs and labels only) |
| `results/` | Row counts, spatial statistics, hypotheses, predictions, metrics; `deviations.md` lists changes to the plan |
| `docs/` | GitHub Pages website and `theory.md` (derivations) |

## Setup

Python 3.12 or newer (the results here were produced with 3.13 and the pinned versions in `requirements.txt`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data access

Raw survey data are **not** included in this repository: neither licence allows public redistribution. Download them as follows and place them at the paths shown.

| Dataset | Source | Access | Put files in |
|---|---|---|---|
| GEMAS Extended Dataset (BGR) | https://data.europa.eu/data/datasets/0554568e-c60f-4106-b099-839bb30f9c44 | Free download: `https://download.bgr.de/bgr/geochemie/GEMAS_ExtendedDataset/csv/ExtendedDataset_CSV.zip` | unzip into `data/raw/gemas/` |
| LUCAS 2018 Topsoil (ESDAC, JRC) | https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data | Free after registration (request form). **We requested access but it was not granted in time; not used in this project.** | the CSV into `data/raw/lucas/` (optional) |
| WorldClim 2.1, 5 arc-min | https://www.worldclim.org/data/worldclim21.html | Free download: `wc2.1_5m_elev.zip`, `wc2.1_5m_bio.zip` | `wc2.1_5m_elev.tif`, `wc2.1_5m_bio_{1,4,12,15}.tif` into `data/raw/worldclim/` |

The website shows only aggregates, model outputs, and sample locations rounded to 0.1°. The split files list sample IDs and fold labels only.

## Run order

```powershell
python scripts/01_clean.py          # cleaned tables + results/data_quality/row_counts.csv
python scripts/02_spatial_stats.py  # results/spatial_stats/*
python scripts/03_splits.py         # data/splits/*, results/splits/split_summary.csv
python scripts/04_hypotheses.py     # training-only statistics + a-priori predictions (results/hypotheses/)
python scripts/05_baselines.py      # baseline test metrics (first script that reads test pH; results/baselines/)
python scripts/07a_tune.py          # inner-CV tuning per outer fold (results/tuning/)
python scripts/07_models.py         # all nine methods on both splits (results/models/)
python scripts/07b_density.py       # GEMAS thinning experiment (results/density/)
python scripts/07c_errors.py        # error analysis (results/errors/)
python scripts/08_verdicts.py       # decision rule applied to H1-H8 (results/verdicts.md)
python scripts/06_export_site.py    # aggregates for the website (docs/data/)
python tests/test_models.py         # unit checks on synthetic data
python tests/check_site.py          # website check (needs: pip install playwright, and Edge or Chrome)
```

All steps are deterministic (seed 42 in `src/config.py`). Steps 04, 05 and 07 take a few minutes each; 07a about 4 minutes and 07b about 10 minutes on 12 cores. LUCAS steps are skipped with a message because the LUCAS CSV is not present. Re-running step 06 rewrites `docs/data/site.json` with differences only in the 16th decimal place.

Reproduction check: a fresh clone, the downloads above and steps 01 to 05 give the committed Phase 1 files up to floating-point noise (largest difference 4e-7, in the RK (RF) prediction).

## Results summary

Test skill = 1 - RMSE / RMSE of the training mean on the same fold; mean over 5 test folds (fold SD in brackets). Full tables: `results/models/gemas_model_summary.csv`, the website (Section 8) and `results/verdicts.md`.

| Method | Spatial split (held-out regions) | Random split |
|---|---|---|
| RK (linear) | 0.205 (0.078) | 0.253 (0.021) |
| UK | 0.200 (0.061) | 0.244 (0.018) |
| OK | 0.196 (0.060) | 0.245 (0.018) |
| Random Forest | 0.195 (0.054) | 0.243 (0.020) |
| GPR | 0.190 (0.054) | 0.247 (0.017) |
| RK (RF) | 0.172 (0.066) | 0.242 (0.016) |
| LinReg (coords + covariates) | 0.130 (0.067) | 0.148 (0.023) |
| LinReg (coords) | 0.091 (0.057) | 0.102 (0.020) |

On held-out regions the five best methods differ by less than one fold SD, so no single winner can be named. Verdicts (decision rule in `results/hypotheses.md` and `results/deviations.md` D2): **H2** (nugget bound) and **H8** (GPR = OK) supported; **H1, H3, H4, H6, H7** inconclusive (for H7 the correlation between predicted and observed skill is 0.62 across fold means, below the 0.7 target); **H5** not testable (no LUCAS). Details and key numbers are in `results/verdicts.md`.

## Where the results are

| Folder | Content |
|---|---|
| `results/hypotheses/`, `results/hypotheses.md` | Pre-registered predictions (not edited after commit) |
| `results/baselines/` | Mean, linear regressions, Random Forest |
| `results/tuning/` | Chosen settings per fold and the full inner-CV grid |
| `results/models/` | Metrics per fold, summary, interval coverage, predictions without observed pH, a-priori check |
| `results/density/` | Thinning experiment (observed and predicted skill) |
| `results/errors/` | Errors by distance, cell, country, fold, pH range; figures |
| `results/verdicts.csv`, `results/verdicts.md` | Hypothesis verdicts with evidence |
| `results/deviations.md` | D1 (no LUCAS), D2 (rules for multi-part hypotheses) |

## Team and contributions

| Phase | Owner | Work |
|---|---|---|
| Phase 1 | Chaitanya | Data access and cleaning, prediction-time inputs, spatial statistics, splits, theory, pre-registered hypotheses, baselines, website skeleton and theory sections |
| Phase 2 | Saksham | OK, UK, GPR, RK; tuning; final evaluation; sparsity experiment; error analysis; hypothesis verdicts; results pages; README completion; publishing |

Chaitanya's work is in pull requests #1 to #8 (`chaitanya/*` branches). Saksham's work is in pull requests #9 to #16 (`saksham/*` branches): models (#9), tuning (#10), evaluation (#11), density experiment (#12), error analysis (#13), verdicts (#14), website sections 7 to 10 (#15) and this README (#16). Both partners review each other's pull requests and are expected to explain every step in the viva.

## AI-use disclosure

**Draft: the students must check and confirm this wording before submission.**

An AI coding assistant (Kiro) was used during Phase 2 for drafting and debugging code (the kriging, tuning, evaluation, density and error-analysis scripts, and the website JavaScript), for editing the website and README text, and for running the headless-browser check of the website. The students defined the research question, the data and the hypotheses, and the pre-registered hypotheses were fixed before any model was evaluated. The students reviewed and ran all code, checked the numbers against the output files, and checked the references against their DOIs or publisher pages. The assistant is a tool and is not an author or contributor of this project.

## Citations

Data:
- GEMAS: Reimann, C., Birke, M., Demetriades, A., Filzmoser, P., & O'Connor, P. (Eds.) (2014). *Chemistry of Europe's Agricultural Soils, Part A*. Geologisches Jahrbuch Reihe B 102. Schweizerbart. Data source: GEMAS Extended Dataset © BGR, Hannover.
- LUCAS: Orgiazzi, A., Ballabio, C., Panagos, P., Jones, A., & Fernández-Ugalde, O. (2018). LUCAS Soil, the largest expandable soil dataset for Europe: a review. *European Journal of Soil Science*, 69(1), 140–153. doi:10.1111/ejss.12499
- LUCAS 2018: Fernández-Ugalde, O., Scarpa, S., Orgiazzi, A., Panagos, P., Van Liedekerke, M., Maréchal, A., & Jones, A. (2022). *LUCAS 2018 Soil Module*. Publications Office of the EU. doi:10.2760/215013
- WorldClim: Fick, S. E., & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4302–4315. doi:10.1002/joc.5086

Methods: see `docs/theory.md` for the full list (Matheron 1963; Cressie 1985, 1993; Webster & Oliver 2007; Stein 1999; Rasmussen & Williams 2006; Hengl et al. 2007; Roberts et al. 2017; Ploton et al. 2020; Wadoux et al. 2021; Tobler 1970).

Additional methods references: Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32. doi:10.1023/A:1010933404324. Mueller, S., Schueler, L., Zech, A., & Hesse, F. (2022). GSTools v1.3: a toolbox for geostatistical modelling in Python. *Geoscientific Model Development*, 15, 3161-3182. doi:10.5194/gmd-15-3161-2022.

Software: NumPy, pandas, SciPy, scikit-learn, GSTools, PyKrige, PySAL (libpysal, esda), rasterio, pyproj, Matplotlib; website: Plotly.js, Leaflet, KaTeX, OpenStreetMap tiles.
