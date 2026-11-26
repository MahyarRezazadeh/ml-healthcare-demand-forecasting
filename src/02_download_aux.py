"""Auxiliary public data: NOAA nClimGrid-daily county weather, HHS state daily series, FEMA fire declarations, ACS county covariates."""
import os, json, importlib.util, requests
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dl", os.path.join(here, "01_download_us.py"))
dl = importlib.util.module_from_spec(spec); spec.loader.exec_module(dl)
RAW = dl.RAW

# 1. weather
for y in range(2020, 2025):
    for m in range(1, 13):
        if (y == 2020 and m < 7) or (y == 2024 and m > 5):
            continue
        for v in ("tmax", "tmin", "prcp"):
            u = f"https://www.ncei.noaa.gov/data/nclimgrid-daily/access/averages/{y}/{v}-{y}{m:02d}-cty-scaled.csv"
            st = dl.fetch(u, f"{RAW}/nclimgrid/{v}-{y}{m:02d}-cty-scaled.csv")
            if st not in ("cached", 200):
                print("weather", v, y, m, st, flush=True)
print("weather done", flush=True)

# 2. HHS state daily
cols = ("state,date,inpatient_beds,inpatient_beds_used,inpatient_beds_used_covid,total_staffed_adult_icu_beds,"
        "staffed_adult_icu_bed_occupancy,previous_day_admission_adult_covid_confirmed,"
        "previous_day_admission_pediatric_covid_confirmed,previous_day_admission_influenza_confirmed,"
        "inpatient_beds_coverage,inpatient_beds_used_coverage,hospital_onset_covid,"
        "all_pediatric_inpatient_bed_occupied,all_pediatric_inpatient_beds,critical_staffing_shortage_today_yes,"
        "critical_staffing_shortage_today_no,critical_staffing_shortage_today_not_reported,deaths_covid")
print("state", dl.fetch(f"https://healthdata.gov/resource/g62h-syeh.csv?$select={cols}&$order=state,date&$limit=200000",
                        f"{RAW}/hhs/state_daily.csv"), flush=True)

# 3. FEMA fire declarations
out, skip = [], 0
p = f"{RAW}/fema/fire_declarations.json"
if not os.path.exists(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    while True:
        u = ("https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?$filter=incidentType%20eq%20'Fire'"
             f"%20and%20fyDeclared%20ge%202005&$top=1000&$skip={skip}")
        r = requests.get(u, timeout=120); r.raise_for_status()
        d = r.json()["DisasterDeclarationsSummaries"]
        out += d; skip += 1000
        if len(d) < 1000:
            break
    json.dump(out, open(p, "w"))
    dl.log("https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries (incidentType=Fire, FY>=2005)", 200, os.path.getsize(p), p, f"{len(out)} rows")
print("fema done", flush=True)

# 4. ACS 2015-2019 5-year county profile
p = f"{RAW}/acs/acs2019_county.json"
if not os.path.exists(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    u = ("https://api.census.gov/data/2019/acs/acs5/profile?get=NAME,DP05_0001E,DP05_0024PE,DP03_0062E,DP03_0128PE,"
         "DP03_0099PE,DP05_0065PE,DP05_0071PE,DP02_0068PE&for=county:*")
    r = requests.get(u, timeout=120)
    open(p, "w", encoding="utf-8").write(r.text)
    dl.log(u, r.status_code, len(r.text), p, "pop, pct65+, med hh income, poverty, uninsured, black, hispanic, BA+")
print("acs done", flush=True)
