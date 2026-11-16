# Can a dataset's statistics predict the best kriging method? Soil pH in unseen European regions

ES 335 course project by **Chaitanya** and **Saksham**.

- **Website:** https://chaitanyachaudhary30.github.io/ES-335-Kriging-Methods/
- **Question:** Can statistics of the training data (sample size, spacing, nugget ratio, range, trend strength) predict which kriging-based method gives the lowest error when predicting topsoil pH (CaCl$_2$) in geographically held-out regions?
- **Unit of analysis:** one row = one georeferenced agricultural topsoil sample (0–20 cm) with its measured pH.

> Status: Phase 1 (data, statistics, splits, theory, hypotheses, baselines, website skeleton) is in progress. Phase 2 (kriging models, experiments, results pages) follows from the `phase1-handoff` tag.

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
| `scripts/01_clean.py` … `05_baselines.py` | Pipeline steps, run in order |
| `data/splits/` | Exact fold assignment for every sample (IDs and labels only) |
| `results/` | Row counts, spatial statistics, hypotheses, predictions, metrics |
| `docs/` | GitHub Pages website and `theory.md` (derivations) |

## Setup

Python 3.12.

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
| LUCAS 2018 Topsoil (ESDAC, JRC) | https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data | Free after registration (request form) | the CSV into `data/raw/lucas/` |
| WorldClim 2.1, 5 arc-min | https://www.worldclim.org/data/worldclim21.html | Free download: `wc2.1_5m_elev.zip`, `wc2.1_5m_bio.zip` | `wc2.1_5m_elev.tif`, `wc2.1_5m_bio_{1,4,12,15}.tif` into `data/raw/worldclim/` |

The website shows only aggregates, model outputs, and sample locations rounded to 0.1°. The split files list sample IDs and fold labels only.

## Run order

```powershell
python scripts/01_clean.py          # cleaned tables + results/data_quality/row_counts.csv
python scripts/02_spatial_stats.py  # results/spatial_stats/*
python scripts/03_splits.py         # data/splits/*, results/splits/split_summary.csv
python scripts/04_hypotheses.py     # training-only statistics + a-priori predictions
python scripts/05_baselines.py      # baseline test metrics (first script that reads test pH)
```

All steps are deterministic (seed 42 in `src/config.py`). LUCAS steps are skipped with a message until the LUCAS CSV is present.

## Team and contributions

| Phase | Owner | Work |
|---|---|---|
| Phase 1 | Chaitanya | Data access and cleaning, prediction-time inputs, spatial statistics, splits, theory, pre-registered hypotheses, baselines, website skeleton and theory sections |
| Phase 2 | Saksham | OK, UK, GPR, RK; tuning; final evaluation; sparsity experiment; error analysis; hypothesis verdicts; results pages; README completion; publishing |

Chaitanya's work is pushed on `chaitanya/*` branches and merged by Saksham; Phase 2 starts from the `phase1-handoff` tag.

## AI-use disclosure

To be completed before submission: which AI tools were used, for which tasks, and how their output was checked.

## Citations

Data:
- GEMAS: Reimann, C., Birke, M., Demetriades, A., Filzmoser, P., & O'Connor, P. (Eds.) (2014). *Chemistry of Europe's Agricultural Soils, Part A*. Geologisches Jahrbuch Reihe B 102. Schweizerbart. Data source: GEMAS Extended Dataset © BGR, Hannover.
- LUCAS: Orgiazzi, A., Ballabio, C., Panagos, P., Jones, A., & Fernández-Ugalde, O. (2018). LUCAS Soil, the largest expandable soil dataset for Europe: a review. *European Journal of Soil Science*, 69(1), 140–153. doi:10.1111/ejss.12499
- LUCAS 2018: Fernández-Ugalde, O., Scarpa, S., Orgiazzi, A., Panagos, P., Van Liedekerke, M., Maréchal, A., & Jones, A. (2022). *LUCAS 2018 Soil Module*. Publications Office of the EU. doi:10.2760/215013
- WorldClim: Fick, S. E., & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4302–4315. doi:10.1002/joc.5086

Methods: see `docs/theory.md` for the full list (Matheron 1963; Cressie 1985, 1993; Webster & Oliver 2007; Stein 1999; Rasmussen & Williams 2006; Hengl et al. 2007; Roberts et al. 2017; Ploton et al. 2020; Wadoux et al. 2021; Tobler 1970).

Software: NumPy, pandas, SciPy, scikit-learn, GSTools, PyKrige, PySAL (libpysal, esda), rasterio, pyproj, Matplotlib; website: Plotly.js, Leaflet, KaTeX, OpenStreetMap tiles.
