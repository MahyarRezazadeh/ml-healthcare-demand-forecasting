"""Context QC-D: Quebec ED-day panel from the archived MSSS hourly file + NAPS PM2.5 + HMS plumes.
Usage: python 09_build_quebec_panel.py step1   (daily outcomes + point files)
       python 09_build_quebec_panel.py step2   (after HMS overlay: exposure + panel)
"""
import sys, numpy as np, pandas as pd
from config import *
Q = f"{RAW}/canada"
step = sys.argv[1]

if step == "step1":
    use = ["No_permis_installation", "RSS", "civieres_fonctionnelles", "civieres_occupees", "patients_civiere_plus24h", "patients_civiere_plus48h",
           "patients_presents_total", "patients_attente_PEC", "DMS_sur_civiere", "DMS_ambulatoire", "mise_a_jour", "commit_time_utc"]
    agg = []
    for ch in pd.read_csv(f"{Q}/quebec/thirdparty_jlomako_qc_er_hourly_panel/qc_er_hourly_panel_2022_2025.csv.gz", usecols=use, chunksize=400000, low_memory=False):
        for c in use[2:10]:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch["ts"] = pd.to_datetime(ch.mise_a_jour, errors="coerce")
        ch = ch.dropna(subset=["ts", "No_permis_installation"]).drop_duplicates(["No_permis_installation", "ts"])
        ch["date"] = ch.ts.dt.normalize()
        ch["occ_rate"] = 100 * ch.civieres_occupees / ch.civieres_fonctionnelles.where(ch.civieres_fonctionnelles > 0)
        g = ch.groupby(["No_permis_installation", "date"]).agg(n=("ts", "size"), occ_sum=("occ_rate", "sum"), occ_n=("occ_rate", "count"), occ_max=("occ_rate", "max"),
              occd_sum=("civieres_occupees", "sum"), func_sum=("civieres_fonctionnelles", "sum"), p24_sum=("patients_civiere_plus24h", "sum"), p24_n=("patients_civiere_plus24h", "count"),
              p48_sum=("patients_civiere_plus48h", "sum"), pres_sum=("patients_presents_total", "sum"), pres_n=("patients_presents_total", "count"),
              att_sum=("patients_attente_PEC", "sum"), att_n=("patients_attente_PEC", "count"), dms_sum=("DMS_sur_civiere", "sum"), dms_n=("DMS_sur_civiere", "count"),
              rss=("RSS", "first"))
        agg.append(g.reset_index())
    a = pd.concat(agg)
    rss = a.dropna(subset=["rss"]).groupby("No_permis_installation").rss.first()
    a = a.drop(columns="rss").groupby(["No_permis_installation", "date"]).agg({c: ("max" if c == "occ_max" else "sum") for c in a.columns if c not in ("No_permis_installation", "date", "rss")}).reset_index()
    d = pd.DataFrame({"ed_id": a.No_permis_installation.astype(int).astype(str), "date": a.date, "n_hours": a.n,
                      "occ": a.occ_sum / a.occ_n.where(a.occ_n > 0), "occ_max": a.occ_max, "stretchers": a.func_sum / a.n, "occupied": a.occd_sum / a.n,
                      "p24": a.p24_sum / a.p24_n.where(a.p24_n > 0), "p48": a.p48_sum / a.p24_n.where(a.p24_n > 0),
                      "present": a.pres_sum / a.pres_n.where(a.pres_n > 0), "waiting": a.att_sum / a.att_n.where(a.att_n > 0), "dms": a.dms_sum / a.dms_n.where(a.dms_n > 0)})
    d["rss"] = d.ed_id.map(rss.rename(index=lambda x: str(int(x))))
    d = d[d.n_hours >= 12]
    inst = pd.read_csv(f"{Q}/quebec/m02_installations.csv", encoding="utf-8-sig", dtype=str, sep=None, engine="python")
    inst = inst.drop_duplicates("INSTAL_COD")[["INSTAL_COD", "LATITUDE", "LONGITUDE", "RSS_CODE", "RSS_NOM"]]
    inst["INSTAL_COD"] = inst.INSTAL_COD.str.lstrip("0")
    d = d.merge(inst, left_on="ed_id", right_on="INSTAL_COD", how="left")
    d["lat"] = pd.to_numeric(d.LATITUDE.str.replace(",", "."), errors="coerce"); d["lon"] = pd.to_numeric(d.LONGITUDE.str.replace(",", "."), errors="coerce")
    d["region"] = d.RSS_CODE.fillna(d.rss.astype(str))
    d.drop(columns=["INSTAL_COD", "LATITUDE", "LONGITUDE", "RSS_CODE", "rss"]).to_parquet(f"{PROC}/qc_ed_day.parquet", index=False)
    print("ED-days", len(d), "EDs", d.ed_id.nunique(), "with coords", d.dropna(subset=["lat"]).ed_id.nunique(), d.date.min(), d.date.max())
    print(d[["occ", "p24", "present", "waiting", "dms", "stretchers"]].describe().round(1).T)
    print(d.groupby(d.date.dt.to_period("Q")).agg(eds=("ed_id", "nunique"), days=("date", "nunique"), occ=("occ", "mean")).to_string())
    d.dropna(subset=["lat"]).drop_duplicates("ed_id").rename(columns={"ed_id": "id"})[["id", "lat", "lon"]].to_parquet(f"{PROC}/pts_qc_ed.parquet", index=False)
    # NAPS
    L = []
    for y in (2019, 2022, 2023):
        n = pd.read_csv(f"{Q}/naps/PM25_{y}.csv", skiprows=7, encoding="utf-8-sig", dtype={2: str})
        n.columns = ["pol", "method", "naps", "city", "prov", "lat", "lon", "date"] + [f"h{i}" for i in range(1, 25)] + list(n.columns[32:])
        n = n[n.prov.isin(["QC", "ON", "NB"])]
        h = n[[f"h{i}" for i in range(1, 25)]].apply(pd.to_numeric, errors="coerce").where(lambda x: x > -900)
        n["pm"] = h.mean(axis=1).where(h.notna().sum(axis=1) >= 18)
        n["date"] = pd.to_datetime(n.date)
        L.append(n.groupby(["naps", "date"]).agg(pm=("pm", "mean"), lat=("lat", "first"), lon=("lon", "first"), prov=("prov", "first")).reset_index())
    n = pd.concat(L).dropna(subset=["pm"]); n.to_parquet(f"{PROC}/naps_site_day.parquet", index=False)
    n.drop_duplicates("naps").rename(columns={"naps": "id"})[["id", "lat", "lon"]].to_parquet(f"{PROC}/pts_naps.parquet", index=False)
    print("NAPS site-days", len(n), "sites", n.naps.nunique(), n.groupby("prov").naps.nunique().to_dict())

if step == "step2":
    import scipy.sparse as sp
    from sklearn.neighbors import BallTree
    d = pd.read_parquet(f"{PROC}/qc_ed_day.parquet").dropna(subset=["lat"])
    n = pd.read_parquet(f"{PROC}/naps_site_day.parquet")
    hs = pd.read_parquet(f"{PROC}/hms_naps.parquet").rename(columns={"id": "naps"})
    he = pd.read_parquet(f"{PROC}/hms_qc_ed.parquet").rename(columns={"id": "ed_id"})
    n = n.merge(hs, on=["naps", "date"], how="left"); n["plume"] = n.dens.notna().astype(int); n["mo"] = n.date.dt.month; n["yr"] = n.date.dt.year
    L = []
    for y in (2022, 2023):   # background: station-month median on non-plume days in the other available years
        b = n[(n.plume == 0) & (n.yr != y)].groupby(["naps", "mo"]).pm.agg(["median", "size"]).reset_index(); b["yr"] = y; L.append(b)
    n = n.merge(pd.concat(L).rename(columns={"median": "bg", "size": "bg_n"}), on=["naps", "mo", "yr"], how="left")
    n.loc[n.bg_n < 10, "bg"] = np.nan
    n["spm"] = np.where(n.plume == 1, (n.pm - n.bg).clip(lower=0), 0.0); n.loc[n.bg.isna(), "spm"] = np.nan
    n = n[n.yr >= 2022]
    S = n.drop_duplicates("naps")[["naps", "lat", "lon"]].reset_index(drop=True); E = d.drop_duplicates("ed_id")[["ed_id", "lat", "lon"]].reset_index(drop=True)
    tree = BallTree(np.deg2rad(S[["lat", "lon"]].values), metric="haversine")
    idx, dist = tree.query_radius(np.deg2rad(E[["lat", "lon"]].values), r=MONITOR_RADIUS_KM / 6371.0, return_distance=True)
    W = sp.csr_matrix((1 / np.maximum(np.concatenate(dist) * 6371, 1), (np.concatenate([np.full(len(i), k) for k, i in enumerate(idx)]), np.concatenate(idx))), shape=(len(E), len(S)))
    dates = pd.date_range("2022-10-01", "2023-12-31")
    def idw(col):
        M = n.pivot(index="naps", columns="date", values=col).reindex(index=S.naps, columns=dates).values
        num = W @ np.nan_to_num(M); den = W @ (~np.isnan(M)).astype(float)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).ravel()
    x = pd.DataFrame({"ed_id": np.repeat(E.ed_id.values, len(dates)), "date": np.tile(dates, len(E)), "spm": idw("spm"), "pm_total": idw("pm")})
    x = x.merge(he, on=["ed_id", "date"], how="left")
    x["plume_any"] = x.dens.notna().astype(int); x["plume_heavy"] = (x.dens == 3).astype(int); x["plume_medheavy"] = (x.dens >= 2).astype(int)
    x = x.sort_values(["ed_id", "date"]); g = x.groupby("ed_id")
    for v in ("spm", "plume_heavy"):
        for k in range(1, 8): x[f"{v}_lag{k}"] = g[v].shift(k)
        for k in (1, 2, 3): x[f"{v}_lead{k}"] = g[v].shift(-k)
    pre = he[(he.date >= "2010-01-01") & (he.date <= EXP_PRE_END)]
    ex = pd.DataFrame({"hms_heavy_1019": pre[pre.dens == 3].groupby("ed_id").size(), "hms_any_1019": pre.groupby("ed_id").size()}).reindex(E.ed_id).fillna(0)
    ex["E"] = (np.arcsinh(ex.hms_heavy_1019) - np.arcsinh(ex.hms_heavy_1019).mean()) / np.arcsinh(ex.hms_heavy_1019).std()
    P = d[(d.date >= "2022-10-24") & (d.date <= "2023-12-31")].merge(x.drop(columns="dens"), on=["ed_id", "date"], how="left").merge(ex, left_on="ed_id", right_index=True, how="left")
    P.to_parquet(f"{PROC}/qc_panel.parquet", index=False)
    print(P.shape, "EDs", P.ed_id.nunique(), "spm non-missing", P.spm.notna().mean().round(3), "regions", P.region.nunique())
    print("experience (HMS heavy days 2010-19):", ex.hms_heavy_1019.describe().round(1).to_dict())
    print(P.groupby(P.date.dt.to_period("M")).agg(spm=("spm", "mean"), spm_max=("spm", "max"), heavy=("plume_heavy", "mean"), d20=("spm", lambda s: (s >= 20).mean())).round(3).to_string())
