"""Stacked episode event study, permutation placebo, and count of independent smoke episodes (context US-W)."""
import warnings, numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")
from pyfixest.core.demean import demean
from config import *
cols = ["hospital_pk", "week", "state", "division", "hospital_subtype", "occ", "occ_cov", "ed", "ed_cov", "icu_occ", "icu_cov", "w1", "w2", "w4", "E1", "E1_terc", "fema_fire_pm4", "sample", "west"]
P = pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=cols)
P = P[(P.hospital_subtype == "Short Term") & (P.fema_fire_pm4 == 0)].copy(); P["l_ed"] = np.log1p(P.ed)
weeks = np.sort(P.week.unique()); P["t"] = P.week.map({w: i for i, w in enumerate(weeks)})

# ---- independent episodes ----
sw = P.groupby(["state", "t"]).w4.mean().unstack(fill_value=0) >= 0.10        # state-weeks with >=10% of hospitals in a severe week
ep = []
for s, r in sw.iterrows():
    v = r.values.astype(int); starts = np.where((v == 1) & (np.r_[0, v[:-1]] == 0))[0]
    for a in starts:
        b = a
        while b + 1 < len(v) and v[b + 1]: b += 1
        ep.append((s, weeks[a], weeks[b], b - a + 1))
ep = pd.DataFrame(ep, columns=["state", "start", "end", "n_weeks"]); ep["division"] = ep.state.map(DIVISION); ep["sample"] = np.where(ep.start <= pd.Timestamp(DISC_END), "discovery", "confirmation")
ep.to_csv(f"{DESC}/usw_smoke_episodes_state.csv", index=False)
nat = (P.groupby("t").w4.mean() >= 0.02).astype(int).values; nat_ep = int(((nat == 1) & (np.r_[0, nat[:-1]] == 0)).sum())
summ = {"state_episodes_total": len(ep), **{f"state_episodes_{k}": int(v) for k, v in ep["sample"].value_counts().items()}, "states_with_any_episode": int(ep.state.nunique()),
        "national_episodes_(>=2%_hospitals_severe)": nat_ep, "weeks_with_any_severe_hospital": int((P.groupby("t").w4.sum() > 0).sum()),
        "severe_hospital_weeks": int(P.w4.sum()), "severe_hospital_weeks_east": int(P[P.west == 0].w4.sum()), "severe_hospital_weeks_west": int(P[P.west == 1].w4.sum()),
        "hospitals_ever_severe": int(P.groupby("hospital_pk").w4.max().sum())}
pd.Series(summ).to_csv(f"{DESC}/usw_episode_counts.csv"); print(pd.Series(summ).to_string())

# ---- stacked event study ----
W4 = P.pivot(index="hospital_pk", columns="t", values="w4"); hosp_div = P.drop_duplicates("hospital_pk").set_index("hospital_pk").division
E1h = P.drop_duplicates("hospital_pk").set_index("hospital_pk").E1; med = E1h.median()
T = W4.shape[1]; A = W4.fillna(0).values
prev4 = np.zeros_like(A)
for k in range(1, 5): prev4[:, k:] += A[:, :-k]
onset = (A == 1) & (prev4 == 0)
anyw = np.zeros_like(A)                        # any severe week within +/-6 weeks
for k in range(-6, 7):
    if k < 0: anyw[:, :k] += A[:, -k:]
    elif k > 0: anyw[:, k:] += A[:, :-k]
    else: anyw += A
stacks, sid = [], 0
hidx = pd.Series(np.arange(len(W4)), index=W4.index)
for dv in sorted(hosp_div.unique()):
    rows_d = hidx[hosp_div[hosp_div == dv].index.intersection(W4.index)].values
    for c in range(5, T - 6):
        tr = rows_d[onset[rows_d, c]]; ct = rows_d[anyw[rows_d, c] == 0]
        if len(tr) >= 20 and len(ct) >= 20:
            sid += 1
            ids = np.r_[W4.index[tr], W4.index[ct]]
            stacks.append(pd.DataFrame({"hospital_pk": np.repeat(ids, 11), "t": np.tile(np.arange(c - 4, c + 7), len(ids)), "k": np.tile(np.arange(-4, 7), len(ids)),
                                        "treat": np.repeat(np.r_[np.ones(len(tr)), np.zeros(len(ct))], 11), "stk": sid, "c": c}))
S = pd.concat(stacks, ignore_index=True).merge(P[["hospital_pk", "t", "occ", "l_ed", "icu_occ", "state", "occ_cov", "ed_cov", "icu_cov", "E1", "sample"]], on=["hospital_pk", "t"], how="inner")
S["hs"] = S.hospital_pk + "_" + S.stk.astype(str); S["ks"] = S.k.astype(str) + "_" + S.stk.astype(str)
print("stacks", sid, "rows", len(S), "treated hospital-episodes", int(S[S.k == 0].treat.sum()))
res = []
for y, cv in (("occ", "occ_cov"), ("l_ed", "ed_cov"), ("icu_occ", "icu_cov")):
    for grp, d in (("all", S), ("treated_low_experience", S[(S.treat == 0) | (S.E1 <= med)]), ("treated_high_experience", S[(S.treat == 0) | (S.E1 > med)]),
                   ("discovery", S[S["sample"] == "discovery"]), ("confirmation", S[S["sample"] == "confirmation"])):
        d = d[d[cv] >= MIN_COVERAGE].dropna(subset=[y]).copy()
        for k in range(-4, 7):
            if k != -1: d[f"D{k + 4}"] = ((d.k == k) & (d.treat == 1)).astype(float)
        terms = [f"D{k + 4}" for k in range(-4, 7) if k != -1]
        try:
            m = pf.feols(f"{y} ~ {' + '.join(terms)} | hs + ks", data=d[[y, "hs", "ks", "state"] + terms], vcov={"CRV1": "state"}); t = m.tidy()
            for k in range(-4, 7):
                if k != -1:
                    res.append(dict(outcome=y, group=grp, k=k, coef=t.loc[f"D{k + 4}", "Estimate"], se=t.loc[f"D{k + 4}", "Std. Error"], p=t.loc[f"D{k + 4}", "Pr(>|t|)"], n=int(m._N), n_stacks=d.stk.nunique()))
        except Exception as ex:
            res.append(dict(outcome=y, group=grp, k=np.nan, note=repr(ex)[:150]))
ES = pd.DataFrame(res); ES.to_csv(f"{ECON}/usw_stacked_event_study.csv", index=False)
print(ES[ES.group.isin(["all"])].pivot(index="k", columns="outcome", values=["coef", "p"]).round(3).to_string())

# ---- permutation placebo (P1): smoke histories reassigned across hospitals within division, 200 draws, FE-only specification ----
d = P[(P.occ_cov >= MIN_COVERAGE)].dropna(subset=["occ", "w1", "E1"]).copy(); d["w"] = d.w1 / 10
fe = np.column_stack([pd.factorize(d.hospital_pk)[0], pd.factorize(d.division + d.t.astype(str))[0]]).astype(np.uint64)
def est(w):
    R, _ = demean(np.column_stack([d.occ.values, w, w * d.E1.values]).astype(float), fe, np.ones(len(d)), tol=1e-7)
    return np.linalg.lstsq(R[:, 1:], R[:, 0], rcond=None)[0]
b0 = est(d.w.values)
Wm = P.assign(w=P.w1 / 10).pivot(index="hospital_pk", columns="t", values="w"); rng = np.random.default_rng(SEED); draws = []
hd = hosp_div.reindex(Wm.index)
for r in range(200):
    perm = pd.Series(Wm.index, index=Wm.index)
    for dv in hd.unique():
        ids = hd[hd == dv].index.values; perm[ids] = rng.permutation(ids)
    wp = Wm.values[hidx.reindex(perm[d.hospital_pk].values).values if False else Wm.index.get_indexer(perm[d.hospital_pk].values), d.t.values]
    ok = ~np.isnan(wp); wp = np.where(ok, wp, 0.0)
    draws.append(est(wp))
draws = np.array(draws)
perm_out = pd.DataFrame({"term": ["beta", "delta"], "actual": b0, "perm_mean": draws.mean(0), "perm_sd": draws.std(0), "perm_p_two_sided": [(np.abs(draws[:, j]) >= abs(b0[j])).mean() for j in (0, 1)]})
perm_out.to_csv(f"{ROB}/usw_permutation_placebo_occ.csv", index=False); print(perm_out.round(4).to_string())
