"""Download US core public data (cached, resumable). Raw files go to data/raw, or to HDF_RAW_DIR if that environment variable is set.
Usage: python 01_download_us.py [hhs|hms|epa|stanford|all]
"""
import sys, os, csv, time, datetime as dt, io
from concurrent.futures import ThreadPoolExecutor
import requests

RAW = os.environ.get("HDF_RAW_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")).replace("\\", "/")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(PROJ, "logs", "data_download_log.csv"); os.makedirs(os.path.dirname(LOG), exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "research-data-download/1.0"


def log(url, status, nbytes, path, note=""):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "url", "http_status", "bytes", "local_path", "note"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), url, status, nbytes, path, note])


def fetch(url, path, quiet=False, retries=3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k in range(retries):
        try:
            r = S.get(url, stream=True, timeout=120)
            if r.status_code == 404:
                if not quiet:
                    log(url, 404, 0, path, "missing")
                return 404
            r.raise_for_status()
            tmp = path + ".part"
            n = 0
            with open(tmp, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch); n += len(ch)
            os.replace(tmp, path)
            if not quiet:
                log(url, r.status_code, n, path)
            return 200
        except Exception as e:
            time.sleep(3 * (k + 1))
            err = repr(e)
    log(url, "ERR", 0, path, err[:200])
    return "ERR"


HHS_COLS = """hospital_pk,collection_week,state,ccn,hospital_name,city,zip,hospital_subtype,fips_code,is_metro_micro,
total_beds_7_day_avg,inpatient_beds_7_day_avg,inpatient_beds_used_7_day_avg,all_adult_hospital_inpatient_beds_7_day_avg,
all_adult_hospital_inpatient_bed_occupied_7_day_avg,inpatient_beds_used_covid_7_day_avg,total_icu_beds_7_day_avg,
total_staffed_adult_icu_beds_7_day_avg,icu_beds_used_7_day_avg,staffed_adult_icu_bed_occupancy_7_day_avg,
total_patients_hospitalized_confirmed_influenza_7_day_avg,previous_day_total_ed_visits_7_day_sum,
previous_day_covid_ed_visits_7_day_sum,previous_day_admission_adult_covid_confirmed_7_day_sum,
previous_day_admission_pediatric_covid_confirmed_7_day_sum,previous_day_admission_influenza_confirmed_7_day_sum,
all_pediatric_inpatient_bed_occupied_7_day_avg,all_pediatric_inpatient_beds_7_day_avg,
inpatient_beds_used_7_day_coverage,inpatient_beds_7_day_coverage,icu_beds_used_7_day_coverage,
geocoded_hospital_address,is_corrected""".replace("\n", "")


def hhs():
    base = "https://healthdata.gov/resource/anag-cw7u.csv"
    step = 100000
    for off in range(0, 1100000, step):
        p = f"{RAW}/hhs/facility_part_{off:07d}.csv"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        url = f"{base}?$select={HHS_COLS}&$order=:id&$limit={step}&$offset={off}"
        st = fetch(url, p)
        print("hhs", off, st, os.path.getsize(p) if os.path.exists(p) else 0, flush=True)
        if os.path.exists(p) and os.path.getsize(p) < 2000:
            break


def hms():
    d0, d1 = dt.date(2005, 8, 5), dt.date(2024, 5, 31)
    days = [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    def one(d):
        url = (f"https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/"
               f"{d:%Y}/{d:%m}/hms_smoke{d:%Y%m%d}.zip")
        return d, fetch(url, f"{RAW}/hms/{d:%Y}/hms_smoke{d:%Y%m%d}.zip", quiet=True)
    res = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (d, st) in enumerate(ex.map(one, days)):
            res[st] = res.get(st, 0) + 1
            if st not in ("cached", 200):
                with open(f"{RAW}/hms/missing_days.txt", "a") as f:
                    f.write(f"{d},{st}\n")
            if i % 500 == 0:
                print("hms", d, res, flush=True)
    log("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/", "batch",
        0, f"{RAW}/hms", f"daily zips 2005-08-05..2024-05-31 result counts {res}")
    print("hms done", res)


def epa():
    for code in ("88101", "88502"):
        for y in range(2015, 2025):
            url = f"https://aqs.epa.gov/aqsweb/airdata/daily_{code}_{y}.zip"
            print("epa", code, y, fetch(url, f"{RAW}/epa/daily_{code}_{y}.zip"), flush=True)


def stanford():
    for fid, name in ((8550336, "smokePM2pt5_predictions_daily_county_20060101-20201231.csv"), (8550346, "README.txt")):
        print("stanford", name, fetch(f"https://dataverse.harvard.edu/api/access/datafile/{fid}", f"{RAW}/stanford/{name}"), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("stanford", stanford), ("hhs", hhs), ("epa", epa), ("hms", hms)):
        if what in (name, "all"):
            fn()
