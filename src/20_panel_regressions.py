"""Panel estimates for context US-W (protocol section 7).
Usage: python 20_panel_regressions.py discovery|confirmation|pooled
Every estimated coefficient is appended to results/causal/usw_<sample>_results.csv, significant or not.
"""
import sys, warnings, numpy as np, pandas as pd, pyfixest as pf
from config import *
warnings.filterwarnings("ignore")
SAMPLE = sys.argv[1]
STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"     # main | altexp | robust | dl | all  (staged to fit 8 GB RAM)
import pyarrow.parquet as pq, gc, re
_all = pq.ParquetFile(f"{PROC}/analysis_panel.parquet").schema.names
_keep = [c for c in _all if re.match(r"^(w[1-5]|pm_total|E\d|E_hms|E1_terc)", c) or c in (
    "hospital_pk", "week", "state", "division", "west", "fips", "hospital_subtype", "is_metro_micro", "occ", "icu_occ", "ed", "ed_noncovid",
    "beds_used", "beds_used_noncovid", "beds", "icu_beds", "adult_occ", "ped_used", "covid_share", "occ_cov", "ed_cov", "icu_cov",
    "tmax", "prcp", "med_income", "share65", "places_copd", "fema_fire_pm4", "nearest_mon_km", "sample")]
P = pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=_keep)
for c in P.select_dtypes("float64").columns:
    P[c] = P[c].astype("float32")
P["divweek"] = P.division + "_" + P.week.dt.strftime("%Y%m%d")
P["stmonth"] = P.state + "_" + P.week.dt.month.astype(str)
P["stweek"] = P.state + "_" + P.week.dt.strftime("%Y%m%d")
P["wk"] = P.week.dt.strftime("%Y%m%d")
P["tbin"] = pd.qcut(P.tmax, 5, labels=False)
for v in [c for c in P.columns if c.startswith(("w1", "w5")) or c == "pm_total"]:
    P[v] = P[v] / 10.0                      # per 10 ug/m3
P["l_ed"] = np.log1p(P.ed); P["l_ed_nc"] = np.log1p(P.ed_noncovid)
P["l_used"] = np.log(P.beds_used.where(P.beds_used > 0)); P["l_used_nc"] = np.log1p(P.beds_used_noncovid)
P["l_beds"] = np.log(P.beds.where(P.beds > 0)); P["l_icu_beds"] = np.log(P.icu_beds.where(P.icu_beds > 0)); P["l_ped"] = np.log1p(P.ped_used)
P["covid_share_l1"] = P.groupby("hospital_pk").covid_share.shift(1)
# baseline (first 12 window weeks) hospital characteristics for interaction controls
b = P[P.week <= "2020-10-24"].groupby("hospital_pk").agg(base_occ=("occ", "mean"), base_beds=("beds", "mean")).reset_index()
P = P.merge(b, on="hospital_pk", how="left")
P["metro"] = (P.is_metro_micro.astype(str).str.lower() == "true").astype(int)
H1 = P.drop_duplicates("hospital_pk")
for c in ("base_occ", "med_income", "share65", "places_copd"):
    P["z_" + c] = (P[c] - H1[c].mean()) / H1[c].std()
P["z_lbeds"] = (np.log(P.base_beds.where(P.base_beds > 0)) - np.log(H1.base_beds.where(H1.base_beds > 0)).mean()) / np.log(H1.base_beds.where(H1.base_beds > 0)).std()

if SAMPLE != "pooled":
    P = P[P["sample"] == SAMPLE]
MAIN = P[(P.hospital_subtype == "Short Term") & (P.fema_fire_pm4 == 0)]
COV = {"occ": "occ_cov", "icu_occ": "icu_cov", "l_ed": "ed_cov", "l_ed_nc": "ed_cov", "ed": "ed_cov"}
PRIMARY = ["occ", "l_ed", "icu_occ"]
SECOND = ["l_used", "l_used_nc", "l_ed_nc", "adult_occ", "l_ped", "l_beds", "l_icu_beds"]
FE = "hospital_pk + divweek + stmonth"
CTRL = "C(tbin) + prcp"
VC = {"CRV1": "state + wk"}
rows = []

def run(y, w, e="E1", data=None, fe=FE, ctrl=CTRL, vc=VC, spec="main", extra="", pois=False, fam="main"):
    d = MAIN if data is None else data
    d = d[d[COV.get(y, "occ_cov")] >= MIN_COVERAGE]
    need = [y, w, "tbin", "prcp"] + ([e] if e else [])
    d = d.dropna(subset=need)
    used = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", f"{y} {w} {e or ''} {ctrl} {extra} {fe} " + " ".join(vc.values()))) | {"hospital_pk", "state", "wk"}
    d = d[[c for c in d.columns if c in used]]
    rhs = f"{w}" + (f" + {w}:{e}" if e else "") + (f" + {e}" if e == "E7" else "") + (f" + {ctrl}" if ctrl else "") + extra
    try:
        m = (pf.fepois if pois else pf.feols)(f"{y} ~ {rhs} | {fe}", data=d, vcov=vc)
        t = m.tidy(); ci = m.confint()
        for term, lab in ((w, "beta"), (f"{w}:{e}", "delta")):
            if term in t.index:
                rows.append(dict(context="US-W", sample=SAMPLE, family=fam, spec=spec, outcome=y, exposure=w, experience=e, term=lab,
                                 coef=t.loc[term, "Estimate"], se=t.loc[term, "Std. Error"], p=t.loc[term, "Pr(>|t|)"],
                                 ci_lo=ci.loc[term].iloc[0], ci_hi=ci.loc[term].iloc[1], n=int(m._N), n_hosp=d.hospital_pk.nunique(),
                                 n_states=d.state.nunique(), n_weeks=d.wk.nunique(), y_mean=d[y].mean(), y_sd=d[y].std(), w_sd=d[w].std()))
        del m; gc.collect()
    except Exception as ex:
        rows.append(dict(context="US-W", sample=SAMPLE, family=fam, spec=spec, outcome=y, exposure=w, experience=e, term="FAILED", note=repr(ex)[:200]))

# 1. main: primary and secondary outcomes x exposure definitions
for y in (PRIMARY + SECOND if STAGE in ('main','all') else []):
    for w in ("w1", "w2", "w3", "w4", "w5"):
        run(y, w, fam="primary" if (y in PRIMARY and w == "w1") else ("main_alt_exposure" if y in PRIMARY else "secondary"))
    run(y, w if False else "w1", e=None, spec="no_interaction", fam="avg_effect")
    run(y, "w2", e=None, spec="no_interaction", fam="avg_effect")
    run(y, "pm_total", e=None, spec="total_pm_no_interaction", fam="avg_effect")
print("main done", len(rows), flush=True)

# 2. alternative experience measures
for y in (PRIMARY if STAGE in ('altexp','all') else []):
    for e in ("E2", "E3", "E4_1y", "E4_3y", "E4_5y", "E4_10y", "E5", "E6", "E7", "E_hms"):
        run(y, "w1", e=e, spec=f"exp_{e}", fam="alt_experience")
    for tname, cond in (("terc", None),):
        d = MAIN.assign(T2=(MAIN.E1_terc == 2).astype(int), T3=(MAIN.E1_terc == 3).astype(int))
        d = d[d[COV[y]] >= MIN_COVERAGE].dropna(subset=[y, "w1", "tbin", "prcp"])
        try:
            m = pf.feols(f"{y} ~ w1 + w1:T2 + w1:T3 + {CTRL} | {FE}", data=d, vcov=VC); t = m.tidy(); ci = m.confint()
            for term in ("w1", "w1:T2", "w1:T3"):
                rows.append(dict(context="US-W", sample=SAMPLE, family="alt_experience", spec="E1_terciles", outcome=y, exposure="w1", experience="E1_terc", term=term,
                                 coef=t.loc[term, "Estimate"], se=t.loc[term, "Std. Error"], p=t.loc[term, "Pr(>|t|)"], ci_lo=ci.loc[term].iloc[0], ci_hi=ci.loc[term].iloc[1], n=int(m._N)))
        except Exception as ex:
            print("terc fail", ex)

# 3. alternative FE / clustering / controls / samples
ALT = MAIN
for y in (PRIMARY if STAGE in ('robust','all') else []):
    for w in ("w1", "w2"):
        run(y, w, fe="hospital_pk + wk", spec="fe_hosp_week", fam="robust")
        run(y, w, fe="hospital_pk + stweek", spec="fe_hosp_stateweek", fam="robust")
        run(y, w, fe="hospital_pk + divweek", spec="fe_no_statemonth", fam="robust")
        run(y, w, vc={"CRV1": "state"}, spec="cluster_state", fam="robust")
        run(y, w, vc={"CRV1": "fips"}, spec="cluster_county", fam="robust")
        run(y, w, vc={"CRV1": "divweek"}, spec="cluster_divweek", fam="robust")
        run(y, w, ctrl=CTRL + " + covid_share_l1", spec="ctrl_lag_covid_share", fam="robust")
        run(y, w, ctrl="", spec="no_weather", fam="robust")
        run(y, w, data=P[(P.hospital_subtype == "Short Term")], spec="incl_fema_fire_weeks", fam="robust")
        run(y, w, data=P[P.fema_fire_pm4 == 0], spec="incl_critical_access", fam="robust")
        run(y, w, data=MAIN[MAIN.west == 1], spec="west_only", fam="robust")
        run(y, w, data=MAIN[MAIN.west == 0], spec="east_only", fam="robust")
        run(y, w, data=MAIN[MAIN.nearest_mon_km <= 50], spec="monitor_within_50km", fam="robust")
        # competing explanations: W x baseline characteristics
        ex = f" + {w}:z_lbeds + {w}:z_base_occ + {w}:metro + {w}:west + {w}:z_med_income + {w}:z_share65 + {w}:z_places_copd"
        run(y, w, extra=ex, data=MAIN.dropna(subset=["z_lbeds", "z_base_occ", "z_med_income", "z_share65", "z_places_copd"]), spec="W_x_baseline_chars", fam="robust")
        # placebo: exposure one year ahead
        run(y, f"{w}_lead52", spec="placebo_lead52", fam="placebo")
    run("ed", "w1", pois=True, spec="ppml", fam="robust") if y == "l_ed" else None
    run("ed", "w2", pois=True, spec="ppml", fam="robust") if y == "l_ed" else None
print("robust done", len(rows), flush=True)

# 4. distributed lag with leads (pre-trend, persistence, recovery, displacement)
dl_rows = []
for y in (PRIMARY + ["l_used_nc", "l_ed_nc", "l_beds"] if STAGE in ("dl","all") else []):
    for w in ("w1", "w2"):
        terms = [f"{w}_lead2", f"{w}_lead1", w] + [f"{w}_lag{k}" for k in range(1, 5)]
        d = MAIN[MAIN[COV.get(y, "occ_cov")] >= MIN_COVERAGE].dropna(subset=[y, "tbin", "prcp", "E1"] + terms)[[y, "tbin", "prcp", "E1", "hospital_pk", "divweek", "stmonth", "state", "wk"] + terms]
        for inter in (False, True):
            rhs = " + ".join(terms) + (" + " + " + ".join(f"{t}:E1" for t in terms) if inter else "")
            try:
                m = pf.feols(f"{y} ~ {rhs} + {CTRL} | {FE}", data=d, vcov=VC); t = m.tidy(); V = pd.DataFrame(m._vcov, index=m._coefnames, columns=m._coefnames)
                for term in t.index:
                    if term.startswith(w):
                        dl_rows.append(dict(sample=SAMPLE, outcome=y, exposure=w, interacted=inter, term=term, coef=t.loc[term, "Estimate"], se=t.loc[term, "Std. Error"], p=t.loc[term, "Pr(>|t|)"], n=int(m._N)))
                for lab, tt in (("cum_lag0_4", terms[2:]), ("cum_lag1_4", terms[3:]), ("sum_leads", terms[:2])) + ((("cumE_lag0_4", [f"{x}:E1" for x in terms[2:]]), ("cumE_lag1_4", [f"{x}:E1" for x in terms[3:]]), ("sumE_leads", [f"{x}:E1" for x in terms[:2]])) if inter else ()):
                    est = t.loc[tt, "Estimate"].sum(); se = np.sqrt(V.loc[tt, tt].values.sum())
                    from scipy import stats
                    dl_rows.append(dict(sample=SAMPLE, outcome=y, exposure=w, interacted=inter, term=lab, coef=est, se=se, p=2 * stats.norm.sf(abs(est / se)), n=int(m._N)))
            except Exception as ex:
                dl_rows.append(dict(sample=SAMPLE, outcome=y, exposure=w, interacted=inter, term="FAILED", note=repr(ex)[:200]))
if dl_rows:
    pd.DataFrame(dl_rows).to_csv(f"{ECON}/usw_{SAMPLE}_distributed_lag.csv", index=False)
if not rows:
    sys.exit()
R = pd.DataFrame(rows)
R["mde_80"] = 2.8 * R.se
R.to_csv(f"{ECON}/usw_{SAMPLE}_results_{STAGE}.csv", index=False)
pd.set_option("display.width", 250)
print(R[(R.family.isin(["primary", "avg_effect"])) & R.outcome.isin(PRIMARY)][["spec", "outcome", "exposure", "term", "coef", "se", "p", "n", "n_hosp"]].round(4).to_string())
