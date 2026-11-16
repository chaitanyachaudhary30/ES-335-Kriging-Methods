"""Spatial statistics: nearest-neighbour distances, Moran's I, trend R^2,
empirical variograms and variogram model fitting.

All distances are in km (ETRS89-LAEA).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gstools as gs
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

MODELS = {
    "Exponential": gs.Exponential,
    "Spherical": gs.Spherical,
    "Matern": gs.Matern,
}


def nn_distances(xy: np.ndarray, xy_ref: np.ndarray | None = None) -> np.ndarray:
    """Distance from each point in xy to its nearest point in xy_ref.

    If xy_ref is None, the nearest *other* point in xy is used.
    """
    if xy_ref is None:
        d, _ = cKDTree(xy).query(xy, k=2)
        return d[:, 1]
    d, _ = cKDTree(xy_ref).query(xy, k=1)
    return d


def occupied_area_km2(xy: np.ndarray, cell_km: float = 50.0) -> float:
    """Area of the 50 km grid cells that contain at least one sample.

    Used as the study-area size when computing sampling density.
    """
    cells = np.unique(np.floor(xy / cell_km).astype(int), axis=0)
    return len(cells) * cell_km**2


def morans_i(xy: np.ndarray, z: np.ndarray, k: int = 8, permutations: int = 999, seed: int = 0) -> dict:
    """Global Moran's I with row-standardised k-nearest-neighbour weights."""
    import warnings

    from esda.moran import Moran
    from libpysal.weights import KNN

    with warnings.catch_warnings():
        # Islands (e.g. Cyprus, Ireland) form separate KNN components; this is expected.
        warnings.simplefilter("ignore", UserWarning)
        w = KNN.from_array(xy, k=k)
    w.transform = "r"
    np.random.seed(seed)
    m = Moran(z, w, permutations=permutations)
    return {"I": float(m.I), "expected_I": float(m.EI), "z_norm": float(m.z_norm), "p_sim": float(m.p_sim), "k": k}


def trend_r2(X: np.ndarray, z: np.ndarray) -> float:
    """In-sample R^2 of a linear trend z ~ X."""
    return float(LinearRegression().fit(X, z).score(X, z))


VARIO_EDGES_KM = np.r_[0, 25, 50, 75, 100, np.arange(150, 851, 50)]


def empirical_variogram(xy: np.ndarray, z: np.ndarray, edges: np.ndarray = VARIO_EDGES_KM):
    """Matheron estimator. Bins are 25 km wide up to 100 km, then 50 km up to 850 km.

    850 km is roughly a quarter to a third of the study-area extent; beyond that
    the estimate relies on pairs from opposite edges of Europe only.
    Bins with 30 or fewer pairs are dropped.
    """
    centers, gamma, counts = gs.vario_estimate((xy[:, 0], xy[:, 1]), z, edges, return_counts=True)
    keep = counts > 30
    return centers[keep], gamma[keep], counts[keep]


@dataclass
class VariogramFit:
    model: str
    nugget: float          # C0
    partial_sill: float    # C1
    len_scale: float       # gstools length scale, km
    nu: float | None       # Matern smoothness
    practical_range: float # distance (km) where correlation falls to 0.05
    r2: float              # fit quality on the empirical variogram

    @property
    def sill(self) -> float:
        return self.nugget + self.partial_sill

    @property
    def nugget_ratio(self) -> float:
        return self.nugget / self.sill

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(sill=self.sill, nugget_ratio=self.nugget_ratio)
        return d

    def covmodel(self) -> gs.CovModel:
        kw = dict(dim=2, var=self.partial_sill, len_scale=self.len_scale, nugget=self.nugget)
        if self.model == "Matern":
            kw["nu"] = self.nu
        return MODELS[self.model](**kw)

    def correlation(self, h: np.ndarray) -> np.ndarray:
        """Correlation r(h) of the structured part (excludes the nugget)."""
        return self.covmodel().correlation(np.asarray(h, dtype=float))

    def gamma(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        g = self.nugget + self.partial_sill * (1 - self.correlation(h))
        return np.where(h == 0, 0.0, g)


def _practical_range(model: gs.CovModel, max_km: float) -> float:
    h = np.linspace(0, 20 * max_km, 20001)
    r = model.correlation(h)
    idx = np.argmax(r <= 0.05)
    return float(h[idx]) if r[idx] <= 0.05 else float("inf")


def fit_variogram(centers, gamma, counts, models=("Exponential", "Spherical", "Matern")) -> tuple[VariogramFit, list[VariogramFit]]:
    """Fit candidate models by weighted least squares and return the best by R^2.

    Weights follow Cressie (1985): N(h) / gamma(h)^2, which gives short lags
    (where kriging weights matter most) more influence than plain pair counts.
    Matern smoothness is bounded to [0.2, 3].
    """
    fits = []
    max_km = float(centers.max())
    weights = counts / np.maximum(gamma, 1e-9) ** 2
    var0 = float(np.max(gamma))
    for name in models:
        r2 = None
        for init in ({"var": 0.6 * var0, "len_scale": max_km / 4, "nugget": 0.4 * var0, "default": "current"}, "default"):
            m = MODELS[name](dim=2)
            kw = dict(nugget=True, weights=weights, return_r2=True, init_guess=init, max_eval=20000)
            if name == "Matern":
                kw["nu"] = True
                m.set_arg_bounds(nu=[0.2, 3.0])
            try:
                _, _, r2 = m.fit_variogram(centers, gamma, **kw)
                break
            except RuntimeError:
                continue
        if r2 is None:
            continue
        fits.append(
            VariogramFit(
                model=name,
                nugget=float(m.nugget),
                partial_sill=float(m.var),
                len_scale=float(m.len_scale),
                nu=float(m.nu) if name == "Matern" else None,
                practical_range=_practical_range(m, max_km),
                r2=float(r2),
            )
        )
    best = max(fits, key=lambda f: f.r2)
    return best, fits
