"""Loading, filtering and quality checks for GEMAS and LUCAS 2018.

Every filter step is logged so that the row-count table on the website and
in the README can be regenerated exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from . import config as C

PH_MIN, PH_MAX = 2.5, 10.5  # physically plausible range for pH in CaCl2


class FilterLog:
    """Records the number of rows remaining after each cleaning step."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.rows: list[dict] = []

    def add(self, step: str, df: pd.DataFrame, note: str = "") -> pd.DataFrame:
        self.rows.append({"dataset": self.dataset, "step": step, "rows": len(df), "note": note})
        return df

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self.rows)
        out["removed"] = (-out["rows"].diff()).fillna(0).astype(int)
        return out


def _to_laea_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = Transformer.from_crs(C.CRS_WGS84, C.CRS_LAEA, always_xy=True)
    x, y = tr.transform(lon, lat)
    return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0


def add_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Sample WorldClim grids at each point (nearest cell, WGS84 lon/lat)."""
    pts = list(zip(df["lon"].to_numpy(), df["lat"].to_numpy()))
    for name, fname in C.COVARIATES.items():
        with rasterio.open(C.WORLDCLIM_DIR / fname) as src:
            vals = np.array([v[0] for v in src.sample(pts)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            vals[vals < -1e30] = np.nan
        df[name] = vals
    return df


def load_gemas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GEMAS agricultural (Ap, ploughed 0-20 cm) soils, common countries only."""
    log = FilterLog("GEMAS")
    ident = pd.read_csv(C.GEMAS_DIR / "11 GEMAS Ap - Ident.csv")
    chem = pd.read_csv(C.GEMAS_DIR / "15a GEMAS Ap - pH-CEC-C-N-S-χ-CIA.csv")
    df = ident.merge(chem[["ID", "pH_CaCl2"]], on="ID", how="left", validate="one_to_one")
    log.add("Extended dataset, Ap samples", df, "includes the eastern-Ukraine extension")

    df = df[df["Type"] == "Ap"]
    df["pH_CaCl2"] = pd.to_numeric(df["pH_CaCl2"], errors="coerce")
    df = log.add("pH_CaCl2 present", df[df["pH_CaCl2"].notna()], "eastern-Ukraine samples have no pH")

    df = log.add(
        "pH in plausible range",
        df[df["pH_CaCl2"].between(PH_MIN, PH_MAX)],
        f"{PH_MIN} <= pH <= {PH_MAX}",
    )
    df = log.add(
        "Study area: 26 EU and UK countries",
        df[df["Country"].isin(C.GEMAS_COMMON_COUNTRIES)],
        "drops NOR, SIL, UKR, SRB, BOS, FOM, MON",
    )
    df = log.add(
        "Unique locations",
        df.drop_duplicates(subset=["X_LAEA", "Y_LAEA"]),
        "no exact duplicate coordinates",
    )

    out = pd.DataFrame(
        {
            "id": df["ID"].astype(int).to_numpy(),
            "country": df["Country"].map(C.GEMAS_TO_NUTS0).to_numpy(),
            "lon": df["X_Coo"].to_numpy(float),
            "lat": df["Y_Coo"].to_numpy(float),
            "ph": df["pH_CaCl2"].to_numpy(float),
        }
    )
    out["x_km"], out["y_km"] = _to_laea_km(out["lon"].to_numpy(), out["lat"].to_numpy())

    # Consistency check: our projection must match the LAEA columns supplied by BGR.
    dx = np.abs(out["x_km"].to_numpy() - df["X_LAEA"].to_numpy() / 1000)
    dy = np.abs(out["y_km"].to_numpy() - df["Y_LAEA"].to_numpy() / 1000)
    log.rows[-1]["note"] += f"; max reprojection difference vs supplied LAEA = {max(dx.max(), dy.max()):.2f} km"

    out = add_covariates(out)
    out = log.add(
        "Covariates available",
        out.dropna(subset=list(C.COVARIATES)),
        "WorldClim cell has data (drops coastal points that fall in sea cells)",
    )
    return out.reset_index(drop=True), log.frame()


# LUCAS 2018 column names differ slightly between releases; accept known variants.
_LUCAS_ALIASES = {
    "id": ["POINTID", "POINT_ID", "Point_ID"],
    "ph": ["pH_CaCl2", "pH_CaCl", "pH(CaCl2)"],
    "lat": ["TH_LAT", "GPS_LAT", "LAT"],
    "lon": ["TH_LONG", "GPS_LONG", "LONG", "LON"],
    "lc": ["LC0_Desc", "LC0_DESC", "LC0"],
    "depth": ["Depth", "DEPTH"],
    "country": ["NUTS_0", "NUTS0"],
}


def _pick(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    for c in _LUCAS_ALIASES[key]:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"LUCAS file has no column for '{key}'. Tried {_LUCAS_ALIASES[key]}; columns: {list(df.columns)}")
    return None


def load_lucas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """LUCAS 2018 topsoil, cropland, 0-20 cm, countries also sampled by GEMAS."""
    files = sorted(C.LUCAS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No LUCAS CSV in {C.LUCAS_DIR}. Request the data from "
            "https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data and place the CSV there."
        )
    log = FilterLog("LUCAS")
    raw = pd.read_csv(files[0], low_memory=False)
    df = log.add("LUCAS 2018 topsoil file", raw, files[0].name)

    c_id, c_ph, c_lat, c_lon = (_pick(df, k) for k in ("id", "ph", "lat", "lon"))
    c_lc, c_depth, c_cty = _pick(df, "lc"), _pick(df, "depth", False), _pick(df, "country")

    df = df.copy()
    df["_ph"] = pd.to_numeric(df[c_ph].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    df = log.add("pH_CaCl2 numeric and present", df[df["_ph"].notna()], "drops '<LOD' or blank entries")
    df = log.add("pH in plausible range", df[df["_ph"].between(PH_MIN, PH_MAX)], f"{PH_MIN} <= pH <= {PH_MAX}")
    if c_depth is not None:
        df = log.add("Depth 0-20 cm", df[df[c_depth].astype(str).str.replace(" ", "").str.startswith("0-20")])
    df = log.add(
        "Cropland only",
        df[df[c_lc].astype(str).str.lower().str.startswith("cropland")],
        "matches GEMAS Ap (ploughed arable land)",
    )
    common = set(C.GEMAS_TO_NUTS0.values())
    df = log.add("Countries also covered by GEMAS", df[df[c_cty].isin(common)])
    df = df[pd.to_numeric(df[c_lat], errors="coerce").notna() & pd.to_numeric(df[c_lon], errors="coerce").notna()]
    df = log.add("Coordinates present", df)
    df = log.add("Unique locations", df.drop_duplicates(subset=[c_lat, c_lon]))

    out = pd.DataFrame(
        {
            "id": df[c_id].to_numpy(),
            "country": df[c_cty].to_numpy(),
            "lon": pd.to_numeric(df[c_lon]).to_numpy(float),
            "lat": pd.to_numeric(df[c_lat]).to_numpy(float),
            "ph": df["_ph"].to_numpy(float),
        }
    )
    out["x_km"], out["y_km"] = _to_laea_km(out["lon"].to_numpy(), out["lat"].to_numpy())
    out = add_covariates(out)
    out = log.add("Covariates available", out.dropna(subset=list(C.COVARIATES)), "WorldClim cell has data")
    return out.reset_index(drop=True), log.frame()
