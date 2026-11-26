"""Assemble econometric results with multiple-testing corrections -> results/causal/*, reports/tables/*"""
import glob, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from config import *
R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ECON}/usw_*_results_*.csv") if "usw_all_" not in f], ignore_index=True)
R = R[R.term.isin(["beta", "delta"])].copy()
# where the two-way clustered variance is undefined, fall back to the state-clustered estimate of the same model (flagged)
cs = R[R.spec == "cluster_state"].set_index(["sample", "outcome", "exposure", "term"])[["se", "p", "ci_lo", "ci_hi"]]
cs = cs[~cs.index.duplicated()]
R["se_note"] = "two-way state+week"
for i, r in R[R.se.isna() & (R.spec == "main")].iterrows():
    k = (r["sample"], r.outcome, r.exposure, r.term)
    if k in cs.index:
        R.loc[i, ["se", "p", "ci_lo", "ci_hi"]] = cs.loc[k].values.astype(float); R.loc[i, "se_note"] = "state-clustered (two-way variance undefined)"
def adj(p, method):
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = ~np.isnan(p)
    if ok.sum(): out[ok] = multipletests(p[ok], method=method)[1]
    return out
R["p_holm_confirmatory"] = np.nan; R["p_bh_family"] = np.nan
PRIM = ["occ", "l_ed", "icu_occ"]
for s in R["sample"].unique():
    c = (R["sample"] == s) & (R.spec == "main") & (R.exposure == "w1") & R.outcome.isin(PRIM)
    R.loc[c, "p_holm_confirmatory"] = adj(R.loc[c, "p"], "holm")
    for term in ("beta", "delta"):
        f = (R["sample"] == s) & (R.spec == "main") & R.exposure.isin(["w1", "w2"]) & (R.term == term)
        R.loc[f, "p_bh_family"] = adj(R.loc[f, "p"], "fdr_bh")
R.to_csv(f"{ECON}/usw_all_results_adjusted.csv", index=False)
LAB = {"occ": "Inpatient occupancy (pp)", "l_ed": "log(1+ED visits)", "icu_occ": "ICU occupancy (pp)", "l_used": "log occupied beds", "l_used_nc": "log(1+non-COVID occupied beds)",
       "l_ed_nc": "log(1+non-COVID ED visits)", "adult_occ": "Adult inpatient occupancy (pp)", "l_ped": "log(1+paediatric occupied beds)", "l_beds": "log staffed inpatient beds", "l_icu_beds": "log ICU beds"}
m = R[(R.spec == "main") & R.exposure.isin(["w1", "w2"])]
b = m[m.term == "beta"].set_index(["sample", "outcome", "exposure"]); d = m[m.term == "delta"].set_index(["sample", "outcome", "exposure"])
T3 = pd.DataFrame({"Context": "US-W", "Sample": b.index.get_level_values(0), "Outcome": b.index.get_level_values(1).map(LAB), "Specification": np.where(b.index.get_level_values(2) == "w1", "smoke PM2.5 per 10 ug/m3", "heavy-plume days"),
                   "Current wildfire effect": b.coef.values, "Standard error": b.se.values, "CI low": b.ci_lo.values, "CI high": b.ci_hi.values, "P-value": b.p.values,
                   "Adjusted p-value (BH)": b.p_bh_family.values, "Holm p (confirmatory set)": b.p_holm_confirmatory.values,
                   "Experience interaction": d.coef.reindex(b.index).values, "Interaction SE": d.se.reindex(b.index).values, "Interaction CI low": d.ci_lo.reindex(b.index).values, "Interaction CI high": d.ci_hi.reindex(b.index).values,
                   "Interaction p": d.p.reindex(b.index).values, "Interaction adjusted p (BH)": d.p_bh_family.reindex(b.index).values, "Interaction Holm p": d.p_holm_confirmatory.reindex(b.index).values,
                   "N": b.n.values, "Hospitals": b.n_hosp.values, "MDE (80% power)": b.mde_80.values, "SE type": b.se_note.values})
conf = T3[(T3.Specification.str.startswith("smoke PM")) & T3.Outcome.isin([LAB[x] for x in PRIM])].sort_values(["Outcome", "Sample"])
rob = R[R.family.isin(["robust", "placebo", "alt_experience"]) & R.outcome.isin(PRIM + ["ed"])].pivot_table(index=["outcome", "exposure", "spec", "term"], columns="sample", values=["coef", "se", "p"]).round(4)
rob.columns = [f"{a}_{b}" for a, b in rob.columns]
DL = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ECON}/usw_*_distributed_lag.csv")], ignore_index=True)
ES = pd.read_csv(f"{ECON}/usw_stacked_event_study.csv"); Q = pd.read_csv(f"{ECON}/qcd_results.csv")
qm = Q[(Q.vcov == "region+date") & Q.spec.isin(["same_day", "same_day_x_experience"])].copy()
for sp in qm.spec.unique():
    for tm in qm[qm.spec == sp].term.unique():
        f = (qm.spec == sp) & (qm.term == tm) if False else (qm.spec == sp) & (qm.term.str.contains(":E") == (":E" in tm))
        qm.loc[f, "p_bh_family"] = adj(qm.loc[f, "p"], "fdr_bh")
qm.to_csv(f"{ECON}/qcd_main_adjusted.csv", index=False)
sheets = {"Table3_main": T3, "confirmatory_Holm": conf, "robustness": rob.reset_index(), "distributed_lag": DL, "stacked_event_study": ES, "quebec_main": qm, "quebec_all": Q}
for nm, f in (("dml", "usw_dml_results.csv"), ("dml_heterogeneity", "usw_dml_heterogeneity_blp.csv"), ("causal_forest_cate", "usw_causal_forest_cate_by_experience.csv")):
    try: sheets[nm] = pd.read_csv(f"{ECON}/{f}")
    except Exception: pass
try: sheets["permutation_placebo"] = pd.read_csv(f"{ROB}/usw_permutation_placebo_occ.csv")
except Exception: pass
with pd.ExcelWriter(f"{TAB}/Econometric_Results.xlsx") as w:
    for k, v in sheets.items(): v.to_excel(w, k[:31], index=False)
T3.to_csv(f"{TAB}/table3_econometric_results.csv", index=False)
open(f"{TAB}/table3_confirmatory.tex", "w", encoding="utf-8").write(conf[["Outcome", "Sample", "Current wildfire effect", "Standard error", "P-value", "Holm p (confirmatory set)", "Experience interaction", "Interaction SE", "Interaction p", "Interaction Holm p", "N"]].to_latex(index=False, float_format="%.3f"))
rob.reset_index().to_csv(f"{ROB}/usw_robustness_by_sample.csv", index=False)
pd.set_option("display.width", 260)
print(conf[["Outcome", "Sample", "Current wildfire effect", "Standard error", "P-value", "Holm p (confirmatory set)", "Experience interaction", "Interaction p", "Interaction Holm p", "MDE (80% power)", "N"]].round(4).to_string())
print(qm[["outcome", "exposure", "term", "coef", "se", "p", "p_bh_family"]].round(4).to_string())
