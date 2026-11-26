"""Context QC-D: Quebec ED-day regressions (Amendment 1). Exploratory by pre-classification."""
import warnings, numpy as np, pandas as pd, pyfixest as pf
from scipy import stats
from config import *
warnings.filterwarnings("ignore")
P = pd.read_parquet(f"{PROC}/qc_panel.parquet")
P["d"] = P.date.dt.strftime("%Y%m%d"); P["regmonth"] = P.region.astype(str) + "_" + P.date.dt.strftime("%Y%m")
P["regdow"] = P.ed_id + "_" + P.date.dt.dayofweek.astype(str)
P["spm10"] = P.spm / 10
for k in range(1, 8): P[f"spm10_lag{k}"] = P[f"spm_lag{k}"] / 10
for k in (1, 2, 3): P[f"spm10_lead{k}"] = P[f"spm_lead{k}"] / 10
P["sev"] = (P.spm >= SEVERE_PM).astype(float).where(P.spm.notna())
P["l_present"] = np.log1p(P.present); P["l_waiting"] = np.log1p(P.waiting); P["l_p24"] = np.log1p(P.p24)
P.loc[~P.occ.between(0, 300), "occ"] = np.nan
OUT = ["occ", "occ_max", "l_p24", "l_present", "l_waiting", "dms"]
FE = "ed_id + d + regmonth + regdow"
rows = []
def add(m, terms, **k):
    t = m.tidy(); ci = m.confint()
    for term in terms:
        if term in t.index:
            rows.append(dict(context="QC-D", term=term, coef=t.loc[term, "Estimate"], se=t.loc[term, "Std. Error"], p=t.loc[term, "Pr(>|t|)"],
                             ci_lo=ci.loc[term].iloc[0], ci_hi=ci.loc[term].iloc[1], n=int(m._N), **k))
for y in OUT:
    for w in ("spm10", "plume_heavy", "sev"):
        d = P.dropna(subset=[y, w, "E"])
        for vcn, vc in (("region+date", {"CRV1": "region + d"}), ("date", {"CRV1": "d"}), ("ed", {"CRV1": "ed_id"})):
            try:
                add(pf.feols(f"{y} ~ {w} | {FE}", data=d, vcov=vc), [w], outcome=y, exposure=w, spec="same_day", vcov=vcn, n_ed=d.ed_id.nunique(), y_mean=d[y].mean())
                add(pf.feols(f"{y} ~ {w} + {w}:E | {FE}", data=d, vcov=vc), [w, f"{w}:E"], outcome=y, exposure=w, spec="same_day_x_experience", vcov=vcn, n_ed=d.ed_id.nunique(), y_mean=d[y].mean())
            except Exception as ex:
                rows.append(dict(context="QC-D", outcome=y, exposure=w, spec="FAILED", note=repr(ex)[:150]))
    # distributed lag: 3 leads, lags 0-7
    terms = [f"spm10_lead{k}" for k in (3, 2, 1)] + ["spm10"] + [f"spm10_lag{k}" for k in range(1, 8)]
    d = P.dropna(subset=[y] + terms)
    try:
        m = pf.feols(f"{y} ~ {' + '.join(terms)} | {FE}", data=d, vcov={"CRV1": "region + d"})
        add(m, terms, outcome=y, exposure="spm10", spec="dist_lag", vcov="region+date", n_ed=d.ed_id.nunique(), y_mean=d[y].mean())
        V = pd.DataFrame(m._vcov, index=m._coefnames, columns=m._coefnames); t = m.tidy()
        for lab, tt in (("cum_lag0_7", terms[3:]), ("cum_lag1_7", terms[4:]), ("sum_leads", terms[:3])):
            est = t.loc[tt, "Estimate"].sum(); se = np.sqrt(V.loc[tt, tt].values.sum())
            rows.append(dict(context="QC-D", outcome=y, exposure="spm10", spec="dist_lag", vcov="region+date", term=lab, coef=est, se=se, p=2 * stats.norm.sf(abs(est / se)), n=int(m._N)))
    except Exception as ex:
        rows.append(dict(context="QC-D", outcome=y, exposure="spm10", spec="FAILED_dl", note=repr(ex)[:150]))
    # alternative FE and placebo (exposure shifted +28 days)
    d = P.dropna(subset=[y, "spm10"]).copy(); d["plc"] = d.groupby("ed_id").spm10.shift(-28)
    for spec, f, w in (("fe_ed_date", "ed_id + d", "spm10"), ("fe_no_dow", "ed_id + d + regmonth", "spm10"), ("placebo_lead28", FE, "plc"), ("june2023_excluded", FE, "spm10")):
        dd = d[~d.date.between("2023-06-01", "2023-06-30")] if spec == "june2023_excluded" else d
        try:
            add(pf.feols(f"{y} ~ {w} | {f}", data=dd.dropna(subset=[w]), vcov={"CRV1": "region + d"}), [w], outcome=y, exposure=w, spec=spec, vcov="region+date")
        except Exception as ex:
            pass
R = pd.DataFrame(rows); R.to_csv(f"{ECON}/qcd_results.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R[(R.vcov == "region+date") & R.spec.isin(["same_day", "same_day_x_experience"])][["outcome", "exposure", "spec", "term", "coef", "se", "p", "n"]].round(4).to_string())
print(R[(R.spec == "dist_lag") & R.term.str.startswith(("cum", "sum"))][["outcome", "term", "coef", "se", "p"]].round(4).to_string())
print(R[R.spec.isin(["fe_ed_date", "fe_no_dow", "placebo_lead28", "june2023_excluded"])][["outcome", "spec", "coef", "se", "p"]].round(4).to_string())
print("severe ED-days:", int((P.spm >= SEVERE_PM).sum()), "distinct dates with any severe ED:", P[P.spm >= SEVERE_PM].date.nunique())
