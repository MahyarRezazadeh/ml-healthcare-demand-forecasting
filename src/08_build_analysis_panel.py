"""Build smoke exposure at hospitals and the analysis panel (protocol sections 5-6).
Requires: hms_hosp.parquet, hms_site.parquet (from 06_overlay_smoke_plumes.py), epa_site_day.parquet.
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.neighbors import BallTree
from config import *

H = pd.read_parquet(f"{PROC}/hospitals.parquet")
S = pd.read_parquet(f"{PROC}/epa_sites.parquet")
pm = pd.read_parquet(f"{PROC}/epa_site_day.parquet")[["site", "date", "pm"]]
hs = pd.read_parquet(f"{PROC}/hms_site.parquet").rename(columns={"id": "site"})
hh = pd.read_parquet(f"{PROC}/hms_hosp.parquet").rename(columns={"id": "hospital_pk"})

# ---- site-day smoke PM ----
pm = pm.merge(hs, on=["site", "date"], how="left")
pm["plume"] = pm.dens.notna().astype(int)
pm["yr"], pm["mo"] = pm.date.dt.year, pm.date.dt.month
clean = pm[pm.plume == 0]
bg = []
for y in range(2020, 2025):
    b = clean[clean.yr.between(y - 3, y - 1)].groupby(["site", "mo"]).pm.agg(["median", "size"]).reset_index()
    b["yr"] = y; bg.append(b)
bg = pd.concat(bg).rename(columns={"median": "bg", "size": "bg_n"})
pm = pm[pm.yr >= 2020].merge(bg, on=["site", "mo", "yr"], how="left")
pm.loc[pm.bg_n < 15, "bg"] = np.nan
pm["spm"] = np.where(pm.plume == 1, (pm.pm - pm.bg).clip(lower=0), 0.0)
pm.loc[pm.bg.isna(), "spm"] = np.nan
pm[["site", "date", "pm", "plume", "dens", "bg", "spm"]].to_parquet(f"{PROC}/site_day_smokepm.parquet", index=False)
print("site-days", len(pm), "share with background", pm.bg.notna().mean().round(3), "share plume", pm.plume.mean().round(3))

# ---- hospital-day IDW ----
R = 6371.0
tree = BallTree(np.deg2rad(S[["lat", "lon"]].values), metric="haversine")
idx, dist = tree.query_radius(np.deg2rad(H[["lat", "lon"]].values), r=MONITOR_RADIUS_KM / R, return_distance=True)
rows = np.concatenate([np.full(len(i), k) for k, i in enumerate(idx)]); cols = np.concatenate(idx)
wts = 1.0 / np.maximum(np.concatenate(dist) * R, 1.0)
W = sp.csr_matrix((wts, (rows, cols)), shape=(len(H), len(S)))
H["n_mon_100km"] = np.array([len(i) for i in idx]); H["nearest_mon_km"] = [d.min() * R if len(d) else np.nan for d in dist]
dates = pd.date_range("2020-01-01", "2024-05-31")
def idw(col):
    M = pm.pivot(index="site", columns="date", values=col).reindex(index=S.site, columns=dates).values
    ok = ~np.isnan(M)
    num = W @ np.nan_to_num(M); den = W @ ok.astype(float)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
hd = pd.DataFrame({"hospital_pk": np.repeat(H.hospital_pk.values, len(dates)), "date": np.tile(dates, len(H)),
                   "spm": idw("spm").ravel(), "pm_total": idw("pm").ravel()})
hd = hd.merge(hh, on=["hospital_pk", "date"], how="left")
hd["plume_any"] = hd.dens.notna().astype("int8")
hd["plume_heavy"] = (hd.dens == 3).astype("int8"); hd["plume_medheavy"] = (hd.dens >= 2).astype("int8")
hd.drop(columns="dens").to_parquet(f"{PROC}/hospital_day_exposure.parquet", index=False)

# ---- validation against Stanford county product, calendar 2020 ----
st = pd.read_parquet(f"{PROC}/stanford_county_day.parquet")
v = hd[hd.date.dt.year == 2020].merge(H[["hospital_pk", "fips"]], on="hospital_pk").merge(st, on=["fips", "date"], how="left", suffixes=("", "_stanford"))
v["spm_stanford"] = v.spm_stanford.fillna(0)
vv = v.dropna(subset=["spm"])
val = {"n_hospital_days": len(vv), "pearson": vv.spm.corr(vv.spm_stanford), "spearman": vv.spm.corr(vv.spm_stanford, method="spearman"),
       "pearson_smoke_days_only": vv[vv.spm_stanford > 0].spm.corr(vv[vv.spm_stanford > 0].spm_stanford),
       "mean_own": vv.spm.mean(), "mean_stanford": vv.spm_stanford.mean(),
       "agree_ge20": ((vv.spm >= 20) == (vv.spm_stanford >= 20)).mean(),
       "own_ge20_days": int((vv.spm >= 20).sum()), "stanford_ge20_days": int((vv.spm_stanford >= 20).sum()),
       "both_ge20": int(((vv.spm >= 20) & (vv.spm_stanford >= 20)).sum())}
pd.Series(val).to_json(f"{DESC}/exposure_validation_vs_stanford_2020.json", indent=1); print(pd.Series(val).round(3).to_string())

# ---- weekly (Sunday start) ----
hd["week"] = hd.date - pd.to_timedelta((hd.date.dt.dayofweek + 1) % 7, unit="D")
wk = hd.groupby(["hospital_pk", "week"]).agg(w1=("spm", "mean"), w1_n=("spm", "count"), w5=("spm", "max"), pm_total=("pm_total", "mean"),
        sev_days=("spm", lambda s: (s >= SEVERE_PM).sum()), w2=("plume_heavy", "sum"), w3=("plume_any", "sum"), w_medheavy=("plume_medheavy", "sum")).reset_index()
wk.loc[wk.w1_n < 5, ["w1", "w5", "pm_total", "sev_days"]] = np.nan
wk["w4"] = ((wk.w1 >= 15) | (wk.w2 >= 2)).astype(int)
# leads / lags on the complete hospital x week grid (exposure exists for every week, reported or not)
wk = wk.sort_values(["hospital_pk", "week"])
assert wk.groupby("hospital_pk").week.diff().dropna().dt.days.eq(7).all()
g = wk.groupby("hospital_pk")
for v in ("w1", "w2", "w4"):
    for k in (1, 2):
        wk[f"{v}_lead{k}"] = g[v].shift(-k)
    for k in (1, 2, 3, 4):
        wk[f"{v}_lag{k}"] = g[v].shift(k)
    wk[f"{v}_lead52"] = g[v].shift(-52)
    wk[f"{v}_lag52"] = g[v].shift(52)
wk["w1_cum4"] = g.w1.transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
wk["w3_run"] = g.w3.transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())

# ---- HMS-based history at the hospital (alternative experience) ----
pre = hh[(hh.date >= "2010-01-01") & (hh.date <= EXP_PRE_END)]
H = H.merge(pre[pre.dens == 3].groupby("hospital_pk").size().rename("hms_heavy_days_1019"), on="hospital_pk", how="left")
H = H.merge(pre.groupby("hospital_pk").size().rename("hms_any_days_1019"), on="hospital_pk", how="left").fillna({"hms_heavy_days_1019": 0, "hms_any_days_1019": 0})

# ---- assemble ----
P = pd.read_parquet(f"{PROC}/hhs_panel.parquet")
P = P.merge(wk, on=["hospital_pk", "week"], how="left")
E = pd.read_parquet(f"{PROC}/county_experience.parquet")
P = P.merge(E, on="fips", how="left").merge(pd.read_parquet(f"{PROC}/county_weather_week.parquet"), on=["fips", "week"], how="left")
P = P.merge(pd.read_parquet(f"{PROC}/county_covariates.parquet"), on="fips", how="left")
P = P.merge(pd.read_parquet(f"{PROC}/fema_fire_county_week.parquet"), on=["fips", "week"], how="left").fillna({"fema_fire_pm4": 0})
P = P.merge(H[["hospital_pk", "n_mon_100km", "nearest_mon_km", "hms_heavy_days_1019", "hms_any_days_1019"]], on="hospital_pk", how="left")
P = P.sort_values(["hospital_pk", "week"]).reset_index(drop=True)
# time-varying experience: pre-sample severe days + in-sample heavy-plume days up to week t-8
cum = P.groupby("hospital_pk").w2.transform(lambda s: s.fillna(0).cumsum().shift(8)).fillna(0)
P["E7_raw"] = np.arcsinh(P.sev_days_0619 + cum)
hosp_E = P.drop_duplicates("hospital_pk")
for raw, name in (("E1_raw", "E1"), ("mean_annual_spm_0619", "E2"), ("episodes_0619", "E3"), ("yrs_since_last_sev", "E5"), ("max_spm_0619", "E6"),
                  ("sev_days_last1y", "E4_1y"), ("sev_days_last3y", "E4_3y"), ("sev_days_last5y", "E4_5y"), ("sev_days_last10y", "E4_10y"),
                  ("hms_heavy_days_1019", "E_hms")):
    x = np.arcsinh(P[raw]) if raw.startswith(("sev_days_last", "episodes", "hms_")) else P[raw]
    ref = np.arcsinh(hosp_E[raw]) if raw.startswith(("sev_days_last", "episodes", "hms_")) else hosp_E[raw]
    P[name] = (x - ref.mean()) / ref.std()
P["E7"] = (P.E7_raw - hosp_E.E1_raw.mean()) / hosp_E.E1_raw.std()
P["E1_terc"] = pd.qcut(P.E1.rank(method="first"), 3, labels=[1, 2, 3]).astype(float)
t = hosp_E.E1_raw.quantile([1 / 3, 2 / 3]).values
P["E1_terc"] = np.select([P.E1_raw <= t[0], P.E1_raw <= t[1]], [1, 2], 3)
P["sample"] = np.where(P.week <= DISC_END, "discovery", "confirmation")
P.to_parquet(f"{PROC}/analysis_panel.parquet", index=False)
print(P.shape, "W1 non-missing", P.w1.notna().mean().round(3), "hospitals with monitor<=100km", int((H.n_mon_100km > 0).sum()), "of", len(H))
print(P.groupby("sample")[["w1", "w2", "w3", "w4", "sev_days"]].mean().round(3))
