"""Write the point files (id, lat, lon) that 06_overlay_smoke_plumes.py reads: hospitals and EPA monitor sites.
Output: data/processed/pts_hosp.parquet, pts_site.parquet
"""
import pandas as pd
from config import *
pd.read_parquet(f"{PROC}/hospitals.parquet").rename(columns={"hospital_pk": "id"})[["id", "lat", "lon"]].to_parquet(f"{PROC}/pts_hosp.parquet", index=False)
pd.read_parquet(f"{PROC}/epa_sites.parquet").rename(columns={"site": "id"})[["id", "lat", "lon"]].to_parquet(f"{PROC}/pts_site.parquet", index=False)
print("point files written")
