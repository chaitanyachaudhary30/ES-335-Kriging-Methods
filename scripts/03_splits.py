"""Step 1.6: spatial block folds, random folds and LUCAS sparsity keys.

The block-to-fold mapping is built once on GEMAS and reused for LUCAS, so
both surveys are tested on the same geographic regions. Blocks that occur only
in LUCAS are assigned with the same balancing rule.

Outputs (IDs and labels only, no pH values)
  data/splits/splits_gemas.csv
  data/splits/splits_lucas.csv            (with sparsity keys u_seed0..u_seed4)
  data/splits/block_fold_mapping.json
  results/splits/split_summary.csv
  docs/data/<dataset>_split_points.json   (rounded locations + labels for the map)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.spatial_stats import nn_distances  # noqa: E402
from src.splits import add_sparsity_keys, block_folds, assign_blocks, make_splits  # noqa: E402

SPLIT_DIR = C.ROOT / "data" / "splits"


def summarise(name: str, df: pd.DataFrame, sp: pd.DataFrame) -> list[dict]:
    xy = df[["x_km", "y_km"]].to_numpy()
    rows = []
    for kind in ("spatial_fold", "random_fold"):
        for f in range(C.N_FOLDS):
            te = sp[kind].to_numpy() == f
            d = nn_distances(xy[te], xy[~te])
            rows.append(
                {
                    "dataset": name,
                    "split": kind.replace("_fold", ""),
                    "fold": f,
                    "n_train": int((~te).sum()),
                    "n_test": int(te.sum()),
                    "test_nn_train_km_median": float(np.median(d)),
                    "test_nn_train_km_mean": float(d.mean()),
                    "test_nn_train_km_p90": float(np.percentile(d, 90)),
                }
            )
    return rows


def web_points(df: pd.DataFrame, sp: pd.DataFrame) -> dict:
    """Locations rounded to 0.1 degrees (about 10 km) with fold labels and the
    distance to the nearest training point under each split. No pH values are
    exported, and the rounding means the exact survey sites are not republished.
    Distances are computed from the exact coordinates before rounding."""
    xy = df[["x_km", "y_km"]].to_numpy()
    out = {"lat": df["lat"].round(1).tolist(), "lon": df["lon"].round(1).tolist()}
    for kind in ("spatial_fold", "random_fold"):
        fold = sp[kind].to_numpy()
        dist = np.zeros(len(df))
        for f in range(C.N_FOLDS):
            te = fold == f
            dist[te] = nn_distances(xy[te], xy[~te])
        out[kind] = fold.tolist()
        out[kind.replace("fold", "dist_km")] = dist.round(1).tolist()
    return out


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "splits").mkdir(parents=True, exist_ok=True)

    gemas = pd.read_csv(C.DATA_PROCESSED / "gemas_clean.csv")
    mapping = block_folds(assign_blocks(gemas["x_km"].to_numpy(), gemas["y_km"].to_numpy()))
    sp_g = make_splits(gemas, mapping)
    sp_g.to_csv(SPLIT_DIR / "splits_gemas.csv", index=False)
    rows = summarise("GEMAS", gemas, sp_g)
    (C.DOCS_DATA / "gemas_split_points.json").write_text(json.dumps(web_points(gemas, sp_g)))

    lucas_path = C.DATA_PROCESSED / "lucas_clean.csv"
    if lucas_path.exists():
        lucas = pd.read_csv(lucas_path)
        sp_l = add_sparsity_keys(make_splits(lucas, mapping))
        # Persist any LUCAS-only blocks into the shared mapping.
        for b, f in zip(sp_l["block"], sp_l["spatial_fold"]):
            mapping.setdefault(b, int(f))
        sp_l.to_csv(SPLIT_DIR / "splits_lucas.csv", index=False)
        rows += summarise("LUCAS", lucas, sp_l)
        (C.DOCS_DATA / "lucas_split_points.json").write_text(json.dumps(web_points(lucas, sp_l)))
    else:
        print("LUCAS not cleaned yet: only GEMAS splits written")

    meta = {
        "block_km": C.BLOCK_KM,
        "n_folds": C.N_FOLDS,
        "seed": C.SEED,
        "grid_origin_laea_km": [0, 0],
        "mapping": {k: int(v) for k, v in sorted(mapping.items())},
    }
    (SPLIT_DIR / "block_fold_mapping.json").write_text(json.dumps(meta, indent=1))
    summary = pd.DataFrame(rows)
    summary.to_csv(C.RESULTS / "splits" / "split_summary.csv", index=False)
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
