"""Spatial block folds, random folds, and LUCAS sparsity subsets.

The split files store only sample IDs and fold labels, never pH values, so
they can be committed without redistributing the survey data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def assign_blocks(x_km: np.ndarray, y_km: np.ndarray, block_km: float = C.BLOCK_KM) -> np.ndarray:
    """Square grid blocks anchored at (0, 0) in LAEA, so both surveys share the same grid."""
    bx = np.floor(x_km / block_km).astype(int)
    by = np.floor(y_km / block_km).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(bx, by)])


def block_folds(blocks: np.ndarray, n_folds: int = C.N_FOLDS, seed: int = C.SEED) -> dict[str, int]:
    """Assign whole blocks to folds, balancing the number of samples per fold.

    Blocks are visited in a random order fixed by the seed, and each block goes
    to the fold that currently holds the fewest samples.
    """
    ids, counts = np.unique(blocks, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(ids))
    load = np.zeros(n_folds, dtype=int)
    mapping: dict[str, int] = {}
    for i in order:
        f = int(np.argmin(load))
        mapping[ids[i]] = f
        load[f] += counts[i]
    return mapping


def make_splits(df: pd.DataFrame, block_mapping: dict[str, int] | None = None, seed: int = C.SEED) -> pd.DataFrame:
    """Return id, block, spatial fold and random fold for every row of df.

    If block_mapping is given (e.g. built for another dataset), it is reused and
    blocks missing from it are assigned with the same balancing rule.
    """
    blocks = assign_blocks(df["x_km"].to_numpy(), df["y_km"].to_numpy())
    mapping = block_folds(blocks, seed=seed) if block_mapping is None else dict(block_mapping)
    missing = sorted(set(blocks) - set(mapping))
    if missing:
        extra = block_folds(np.array([b for b in blocks if b in missing]), seed=seed)
        mapping.update(extra)

    rng = np.random.default_rng(seed)
    random_fold = np.arange(len(df)) % C.N_FOLDS
    rng.shuffle(random_fold)

    return pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "block": blocks,
            "spatial_fold": [mapping[b] for b in blocks],
            "random_fold": random_fold,
        }
    )


def add_sparsity_keys(splits: pd.DataFrame, seeds=C.SPARSITY_SEEDS) -> pd.DataFrame:
    """Add one uniform random key per seed.

    For a training fold, the subset at fraction f and seed s is the f * n_train
    training points with the smallest key u_s. Subsets are therefore nested:
    the 10% subset is contained in the 25% subset, and so on, which makes the
    density curves smoother and easier to interpret.
    """
    out = splits.copy()
    for s in seeds:
        out[f"u_seed{s}"] = np.random.default_rng(1000 + s).random(len(out))
    return out


def sparsity_subset(train_idx: np.ndarray, keys: np.ndarray, fraction: float) -> np.ndarray:
    """Indices (subset of train_idx) kept at a given training fraction."""
    k = max(1, int(round(fraction * len(train_idx))))
    order = np.argsort(keys[train_idx], kind="stable")
    return np.sort(train_idx[order[:k]])
