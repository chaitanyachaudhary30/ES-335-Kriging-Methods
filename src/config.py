"""Project-wide settings: paths, projection, seeds and split parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
DOCS_DATA = ROOT / "docs" / "data"

GEMAS_DIR = DATA_RAW / "gemas"
LUCAS_DIR = DATA_RAW / "lucas"
WORLDCLIM_DIR = DATA_RAW / "worldclim"

# All distances are computed in ETRS89-LAEA (EPSG:3035) and reported in km.
CRS_LAEA = "EPSG:3035"
CRS_WGS84 = "EPSG:4326"

SEED = 42
N_FOLDS = 5

# Spatial block side length in km. The fitted GEMAS pH variograms reach 95%
# of their sill at 380-520 km (results/spatial_stats/summary.json), so 500 km
# blocks keep most test points beyond the correlation range of the training
# data on at least one side. The same size is used for both surveys so that
# the test geometry is comparable.
BLOCK_KM = 500

# LUCAS sparsity experiment: training fractions and repeat seeds.
SPARSITY_FRACTIONS = [1.0, 0.5, 0.25, 0.10]
SPARSITY_SEEDS = [0, 1, 2, 3, 4]

# Countries present in both surveys (GEMAS three-letter codes).
# GEMAS also sampled Norway, Switzerland, Ukraine and Western Balkan states,
# which LUCAS 2018 (EU-27 + UK) does not cover. They are excluded so that
# both datasets describe the same region.
GEMAS_COMMON_COUNTRIES = [
    "AUS", "BEL", "BUL", "CRO", "CYP", "CZR", "DEN", "EST", "FIN", "FRA",
    "GER", "HEL", "HUN", "IRL", "ITA", "LAV", "LIT", "LUX", "NEL", "POL",
    "PTG", "SKA", "SLO", "SPA", "SWE", "UNK",
]
# Same countries as LUCAS NUTS_0 codes.
GEMAS_TO_NUTS0 = {
    "AUS": "AT", "BEL": "BE", "BUL": "BG", "CRO": "HR", "CYP": "CY",
    "CZR": "CZ", "DEN": "DK", "EST": "EE", "FIN": "FI", "FRA": "FR",
    "GER": "DE", "HEL": "EL", "HUN": "HU", "IRL": "IE", "ITA": "IT",
    "LAV": "LV", "LIT": "LT", "LUX": "LU", "NEL": "NL", "POL": "PL",
    "PTG": "PT", "SKA": "SK", "SLO": "SI", "SPA": "ES", "SWE": "SE",
    "UNK": "UK",
}

# Covariates: WorldClim 2.1, 5 arc-minute grids. They exist for every
# location in Europe, so they are legitimate inputs at prediction time.
COVARIATES = {
    "elev": "wc2.1_5m_elev.tif",      # elevation, m
    "bio1": "wc2.1_5m_bio_1.tif",     # annual mean temperature, deg C
    "bio4": "wc2.1_5m_bio_4.tif",     # temperature seasonality
    "bio12": "wc2.1_5m_bio_12.tif",   # annual precipitation, mm
    "bio15": "wc2.1_5m_bio_15.tif",   # precipitation seasonality
}
COORDS = ["x_km", "y_km"]
TARGET = "ph"
