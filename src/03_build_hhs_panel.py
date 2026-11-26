"""Clean the HHS facility-week file. No exposure data are touched here.
Output: data/processed/hhs_panel.parquet, hospitals.parquet, and a data-quality summary.
"""
import glob, os, re, numpy as np, pandas as pd
from config import *

parts = sorted(glob.glob(f"{RAW}/hhs/facility_part_*.csv"))
df = pd.concat([pd.read_csv(p, dtype={"hospital_pk": str, "ccn": str, "zip": str, "fips_code": str}) for p in parts], ignore_index=True)
n_raw = len(df)
df["week"] = pd.to_datetime(df["collection_week"]).dt.normalize()
num = [c for c in df.columns if c.endswith(("_avg", "_sum", "_coverage"))]
q = {"raw_rows": n_raw, "raw_hospitals": df.hospital_pk.nunique(), "raw_weeks": df.week.nunique(),
     "week_min": str(df.week.min().date()), "week_max": str(df.week.max().date()),
     "weekday_of_collection_week": df.week.dt.day_name().mode()[0]}
supp = {c: int((df[c] == -999999).sum()) for c in num}
for c in num:
    df[c] = df[c].where(df[c] != -999999)
    df.loc[df[c] < 0, c] = np.nan
q["suppressed_cells_key"] = {k: supp[k] for k in ["inpatient_beds_used_7_day_avg", "icu_beds_used_7_day_avg", "previous_day_total_ed_visits_7_day_sum", "inpatient_beds_used_covid_7_day_avg"]}

# duplicates
dup = df.duplicated(["hospital_pk", "week"]).sum(); q["duplicate_hospital_weeks"] = int(dup)
df = df.sort_values(["hospital_pk", "week", "is_corrected"]).drop_duplicates(["hospital_pk", "week"], keep="last")

# location
m = df["geocoded_hospital_address"].str.extract(r"POINT \(([-\d\.]+) ([-\d\.]+)\)").astype(float)
df["lon"], df["lat"] = m[0], m[1]
df["fips"] = df["fips_code"].str.zfill(5)

# outcomes (definitions in protocol section 4)
df["occ"] = 100 * df.inpatient_beds_used_7_day_avg / df.inpatient_beds_7_day_avg
df["icu_occ"] = 100 * df.icu_beds_used_7_day_avg / df.total_icu_beds_7_day_avg
for c in ("occ", "icu_occ"):
    df.loc[~df[c].between(0, 150), c] = np.nan
df["ed"] = df.previous_day_total_ed_visits_7_day_sum
df["ed_noncovid"] = df.ed - df.previous_day_covid_ed_visits_7_day_sum
df.loc[df.ed_noncovid < 0, "ed_noncovid"] = np.nan
df["beds_used"] = df.inpatient_beds_used_7_day_avg
df["beds_used_noncovid"] = df.beds_used - df.inpatient_beds_used_covid_7_day_avg
df.loc[df.beds_used_noncovid < 0, "beds_used_noncovid"] = np.nan
df["beds"] = df.inpatient_beds_7_day_avg
df["icu_beds"] = df.total_icu_beds_7_day_avg
df["icu_used"] = df.icu_beds_used_7_day_avg
df["adult_occ"] = 100 * df.all_adult_hospital_inpatient_bed_occupied_7_day_avg / df.all_adult_hospital_inpatient_beds_7_day_avg
df.loc[~df.adult_occ.between(0, 150), "adult_occ"] = np.nan
df["ped_used"] = df.all_pediatric_inpatient_bed_occupied_7_day_avg
df["covid_share"] = 100 * df.inpatient_beds_used_covid_7_day_avg / df.inpatient_beds_7_day_avg
df.loc[~df.covid_share.between(0, 150), "covid_share"] = np.nan
df["flu_adm"] = df.previous_day_admission_influenza_confirmed_7_day_sum
df["covid_adm"] = df.previous_day_admission_adult_covid_confirmed_7_day_sum

df = df[(df.week >= WIN_START) & (df.week <= WIN_END)]
q["rows_in_window"] = len(df); q["weeks_in_window"] = int(df.week.nunique())
conus = ~df.state.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"]) & df.state.isin(list(DIVISION))
df = df[conus & df.lat.notna()]
df = df[df.hospital_subtype.isin(["Short Term", "Critical Access Hospitals"])]
df["division"] = df.state.map(DIVISION)
df["west"] = df.division.isin(WEST).astype(int)
nw = df.week.nunique()
cov = df.groupby("hospital_pk").agg(occ_cov=("occ", lambda s: s.notna().sum() / nw), ed_cov=("ed", lambda s: s.notna().sum() / nw),
                                    icu_cov=("icu_occ", lambda s: s.notna().sum() / nw))
q["hospitals_conus_acute"] = int(len(cov))
for c in cov: q[f"hospitals_{c}_ge80"] = int((cov[c] >= MIN_COVERAGE).sum())
df = df.merge(cov, left_on="hospital_pk", right_index=True)
keep = ["hospital_pk", "week", "state", "division", "west", "fips", "zip", "hospital_subtype", "is_metro_micro", "lat", "lon",
        "occ", "icu_occ", "ed", "ed_noncovid", "beds_used", "beds_used_noncovid", "beds", "icu_beds", "icu_used", "adult_occ",
        "ped_used", "covid_share", "flu_adm", "covid_adm", "occ_cov", "ed_cov", "icu_cov"]
df = df[keep].sort_values(["hospital_pk", "week"]).reset_index(drop=True)
df.to_parquet(f"{PROC}/hhs_panel.parquet", index=False)
hosp = df.groupby("hospital_pk").agg(state=("state", "first"), division=("division", "first"), west=("west", "first"), fips=("fips", "first"),
                                     subtype=("hospital_subtype", "first"), metro=("is_metro_micro", "first"),
                                     lat=("lat", "median"), lon=("lon", "median"), occ_cov=("occ_cov", "first"),
                                     ed_cov=("ed_cov", "first"), icu_cov=("icu_cov", "first")).reset_index()
hosp.to_parquet(f"{PROC}/hospitals.parquet", index=False)
q["final_rows"] = len(df); q["final_hospitals"] = len(hosp)
q["missing_share"] = df[["occ", "icu_occ", "ed", "beds_used_noncovid", "ed_noncovid", "ped_used"]].isna().mean().round(4).to_dict()
# reporting over time: share of hospitals reporting each outcome by quarter
rep = df.assign(qtr=df.week.dt.to_period("Q").astype(str)).groupby("qtr")[["occ", "icu_occ", "ed", "ed_noncovid"]].agg(lambda s: s.notna().mean()).round(3)
rep.to_csv(f"{DESC}/hhs_reporting_by_quarter.csv")
pd.Series(q).to_json(f"{DESC}/hhs_quality_summary.json", indent=1)
print(pd.Series(q).to_string()); print(rep.to_string())
