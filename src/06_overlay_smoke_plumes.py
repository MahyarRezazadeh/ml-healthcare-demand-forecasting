"""Overlay NOAA HMS daily smoke polygons on a set of points.
Usage: python 06_overlay_smoke_plumes.py <points.parquet with id,lat,lon> <out.parquet> <start YYYY-MM-DD> <end YYYY-MM-DD>
Output is sparse: one row per point-day under a plume, dens = 1 light, 2 medium, 3 heavy (0 = plume with unknown density).
Also writes <out>_days.csv listing every day with file status (ok / missing / unreadable / n polygons).
"""
import sys, os, warnings, datetime as dt, numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely import STRtree
from config import *
warnings.filterwarnings("ignore")
pts_file, out_file, d0, d1 = sys.argv[1:5]
P = pd.read_parquet(pts_file)[["id", "lat", "lon"]].dropna().reset_index(drop=True)
geom = shapely.points(P.lon.values, P.lat.values)
tree = STRtree(geom)
DMAP = {"light": 1, "medium": 2, "heavy": 3, "5": 1, "16": 2, "27": 3, "5.0": 1, "16.0": 2, "27.0": 3,
        "thin": 1, "thick": 3, "moderate": 2}
rows, status = [], []
day = dt.date.fromisoformat(d0); end = dt.date.fromisoformat(d1)
while day <= end:
    f = f"{RAW}/hms/{day:%Y}/hms_smoke{day:%Y%m%d}.zip"
    if not os.path.exists(f):
        status.append((day, "missing", 0)); day += dt.timedelta(1); continue
    try:
        g = gpd.read_file(f"zip://{f}")
    except Exception as e:
        status.append((day, "unreadable", 0)); day += dt.timedelta(1); continue
    if len(g) == 0:
        status.append((day, "ok", 0)); day += dt.timedelta(1); continue
    dcol = next((c for c in g.columns if c.lower() == "density"), None)
    dens = (g[dcol].astype(str).str.strip().str.lower().map(DMAP).fillna(0).astype(int).values if dcol else np.zeros(len(g), int))
    polys = shapely.make_valid(g.geometry.values)
    best = np.full(len(P), -1, int)
    for poly, dn in zip(polys, dens):
        if poly is None or shapely.is_empty(poly):
            continue
        idx = tree.query(poly, predicate="intersects")
        if len(idx):
            best[idx] = np.maximum(best[idx], dn)
    hit = np.where(best >= 0)[0]
    if len(hit):
        rows.append(pd.DataFrame({"id": P.id.values[hit], "date": pd.Timestamp(day), "dens": best[hit].astype("int8")}))
    status.append((day, "ok", len(g)))
    if day.day == 1 and day.month in (1, 7):
        print(day, len(rows), flush=True)
    day += dt.timedelta(1)
res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["id", "date", "dens"])
res.to_parquet(out_file, index=False)
pd.DataFrame(status, columns=["date", "status", "n_polygons"]).to_csv(out_file.replace(".parquet", "_days.csv"), index=False)
print("done", len(res), pd.DataFrame(status)[1].value_counts().to_dict())
