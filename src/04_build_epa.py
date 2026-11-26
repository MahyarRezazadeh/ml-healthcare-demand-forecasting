"""EPA AQS daily PM2.5 (88101 FRM/FEM and 88502 non-FRM) -> site-day table and site list."""
import glob, zipfile, pandas as pd, numpy as np
from config import *
out = []
for p in sorted(glob.glob(f"{RAW}/epa/daily_*.zip")):
    yr = int(p[-8:-4])
    if yr < 2017:
        continue
    with zipfile.ZipFile(p) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), usecols=["State Code", "County Code", "Site Num", "Latitude", "Longitude",
                        "Sample Duration", "Date Local", "Arithmetic Mean", "Observation Percent", "Event Type"],
                        dtype={"State Code": str, "County Code": str, "Site Num": str}, low_memory=False)
    d = d[d["State Code"].str.isdigit()]
    d = d[d["State Code"].astype(int) <= 56]
    d["site"] = d["State Code"] + d["County Code"] + d["Site Num"]
    d["date"] = pd.to_datetime(d["Date Local"])
    # prefer 24-hour / 24-hr block averages; average across monitors and parameters at a site-day
    g = d.groupby(["site", "date"]).agg(pm=("Arithmetic Mean", "mean"), lat=("Latitude", "first"), lon=("Longitude", "first")).reset_index()
    g["src"] = p.split("daily_")[1][:5]
    out.append(g); print(p[-20:], len(g), flush=True)
a = pd.concat(out, ignore_index=True)
a = a.groupby(["site", "date"]).agg(pm=("pm", "mean"), lat=("lat", "first"), lon=("lon", "first")).reset_index()
a.loc[a.pm < 0, "pm"] = 0
a = a[a.lon.between(-125, -66) & a.lat.between(24, 50)]
a.to_parquet(f"{PROC}/epa_site_day.parquet", index=False)
s = a.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first"), n=("pm", "size"), first=("date", "min"), last=("date", "max")).reset_index()
s.to_parquet(f"{PROC}/epa_sites.parquet", index=False)
print(len(a), len(s), a.date.min(), a.date.max())
