"""County prior-experience measures (Stanford smoke PM 2006-2019), weekly weather, county covariates, FEMA fire weeks."""
import glob, json, numpy as np, pandas as pd
from config import *

# ---------- 1. prior experience, 2006-2019 only ----------
s = pd.read_csv(f"{RAW}/stanford/smokePM2pt5_predictions_daily_county_20060101-20201231.csv", sep="\t", dtype={"GEOID": str, "date": str})
s["date"] = pd.to_datetime(s["date"], format="%Y%m%d")
s = s.rename(columns={"GEOID": "fips", "smokePM_pred": "spm"})
s.to_parquet(f"{PROC}/stanford_county_day.parquet", index=False)
pre = s[s.date <= EXP_PRE_END]
sev = pre[pre.spm >= SEVERE_PM].sort_values(["fips", "date"])
allc = pd.Index(sorted(s.fips.unique()), name="fips")
E = pd.DataFrame(index=allc)
E["sev_days_0619"] = sev.groupby("fips").size()
for k, y0 in ((1, 2019), (3, 2017), (5, 2015), (10, 2010)):
    E[f"sev_days_last{k}y"] = sev[sev.date.dt.year >= y0].groupby("fips").size()
gap = sev.groupby("fips").date.diff().dt.days
sev = sev.assign(new_ep=(gap.isna() | (gap > 7)).astype(int))
E["episodes_0619"] = sev.groupby("fips").new_ep.sum()
E["mean_annual_spm_0619"] = pre.groupby("fips").spm.sum() / (14 * 365.25)
E["max_spm_0619"] = pre.groupby("fips").spm.max()
E["smoke_days_any_0619"] = pre[pre.spm > 0].groupby("fips").size()
for th in (10, 35, 50):
    E[f"days_ge{th}_0619"] = pre[pre.spm >= th].groupby("fips").size()
last = sev.groupby("fips").date.max()
E["yrs_since_last_sev"] = ((pd.Timestamp(EXP_PRE_END) - last).dt.days / 365.25)
E = E.fillna({c: 0 for c in E.columns if c != "yrs_since_last_sev"})
E["yrs_since_last_sev"] = E["yrs_since_last_sev"].fillna(14.0)   # never in window -> censored at window length
E["E1_raw"] = np.arcsinh(E.sev_days_0619)
E.reset_index().to_parquet(f"{PROC}/county_experience.parquet", index=False)
print(E.describe().T[["mean", "50%", "max"]].round(2).to_string())

# ---------- 2. weather: Sunday-start weeks ----------
POSTAL_FIPS = {'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','DC':'11','FL':'12','GA':'13','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
wx = []
for v in ("tmax", "tmin", "prcp"):
    L = []
    for f in sorted(glob.glob(f"{RAW}/nclimgrid/{v}-*-cty-scaled.csv")):
        d = pd.read_csv(f, header=None, dtype={1: str})
        d = d[d[0] == "cty"]
        st = d[2].str[:2].map(POSTAL_FIPS)
        d = d[st.notna()]
        fips = st[st.notna()] + d[1].str[-3:]
        vals = d.iloc[:, 6:37].astype(float).values
        y, m = int(d.iloc[0, 3]), int(d.iloc[0, 4])
        nd = pd.Period(f"{y}-{m:02d}").days_in_month
        long = pd.DataFrame(vals[:, :nd], index=fips.values, columns=pd.date_range(f"{y}-{m:02d}-01", periods=nd)).stack().rename(v).reset_index()
        long.columns = ["fips", "date", v]
        L.append(long)
    wx.append(pd.concat(L).set_index(["fips", "date"]))
w = pd.concat(wx, axis=1).reset_index()
w = w.replace(-999.99, np.nan)
w.to_parquet(f"{PROC}/county_weather_day.parquet", index=False)
w["week"] = w.date - pd.to_timedelta((w.date.dt.dayofweek + 1) % 7, unit="D")
ww = w.groupby(["fips", "week"]).agg(tmax=("tmax", "mean"), tmin=("tmin", "mean"), prcp=("prcp", "sum"), tmax_max=("tmax", "max")).reset_index()
ww.to_parquet(f"{PROC}/county_weather_week.parquet", index=False)
print("weather", len(ww), ww.week.min(), ww.week.max())

# ---------- 3. county covariates (all measured 2019 or earlier except PLACES, see data dictionary) ----------
x = pd.read_excel(f"{RAW}/acs/est19all.xls", header=3, dtype=str)
x["fips"] = x["State FIPS Code"] + x["County FIPS Code"]
c = x[["fips"]].copy()
c["med_income"] = pd.to_numeric(x["Median Household Income"], errors="coerce")
c["poverty"] = pd.to_numeric(x["Poverty Percent, All Ages"], errors="coerce")
a = pd.concat([pd.read_csv(f, dtype={"STATE": str, "COUNTY": str}, encoding="latin1") for f in glob.glob(f"{RAW}/acs/agesex/*.csv")])
a = a[a.YEAR == 12]
a["fips"] = a.STATE + a.COUNTY
a["pop"] = a.POPESTIMATE; a["share65"] = 100 * a.AGE65PLUS_TOT / a.POPESTIMATE; a["share_u18"] = 100 * (a.POPESTIMATE - a.AGE18PLUS_TOT) / a.POPESTIMATE
c = c.merge(a[["fips", "pop", "share65", "share_u18"]], on="fips", how="outer")
p = pd.read_csv(f"{RAW}/acs/places_county.csv", dtype={"locationid": str})
p = p.sort_values("year").drop_duplicates(["locationid", "measureid"], keep="last").pivot(index="locationid", columns="measureid", values="data_value")
p.columns = ["places_" + k.lower() for k in p.columns]
c = c.merge(p, left_on="fips", right_index=True, how="left")
c.to_parquet(f"{PROC}/county_covariates.parquet", index=False)
print("covariates", c.shape, c.notna().mean().round(2).to_dict())

# ---------- 4. FEMA fire declarations -> county-week flags ----------
f = pd.DataFrame(json.load(open(f"{RAW}/fema/fire_declarations.json")))
f["fips"] = f.fipsStateCode + f.fipsCountyCode
f["b"] = pd.to_datetime(f.incidentBeginDate).dt.tz_localize(None).dt.normalize()
f["e"] = pd.to_datetime(f.incidentEndDate).dt.tz_localize(None).dt.normalize().fillna(f["b"] + pd.Timedelta(days=14))
f = f[(f.e >= "2020-06-01") & (f.fipsCountyCode != "000")]
rows = []
for r in f.itertuples():
    for d in pd.date_range(r.b - pd.Timedelta(weeks=4), r.e + pd.Timedelta(weeks=4), freq="W-SUN"):
        rows.append((r.fips, d))
ff = pd.DataFrame(rows, columns=["fips", "week"]).drop_duplicates(); ff["fema_fire_pm4"] = 1
ff.to_parquet(f"{PROC}/fema_fire_county_week.parquet", index=False)
print("fema", len(f), f.declarationType.value_counts().to_dict(), "county-weeks flagged", len(ff))
