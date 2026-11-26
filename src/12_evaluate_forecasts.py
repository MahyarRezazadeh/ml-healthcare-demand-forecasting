"""Evaluate walk-forward forecasts: metrics by regime, R2_OOS, dependence-aware tests, intervals, congestion detection.
Usage: python 12_evaluate_forecasts.py <outcome> <h>
"""
import sys, glob, os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from config import *
outcome, h = sys.argv[1], int(sys.argv[2])
D = f"{ML}/preds_{outcome}_h{h}"; TAG = f"{outcome}_h{h}"
B = pd.read_parquet(f"{D}/_benchmarks.parquet")
key = ["hospital_pk", "week"]
for f in sorted(glob.glob(f"{D}/*.parquet")):
    n = os.path.basename(f)[:-8]
    if n.startswith("_"): continue
    B = B.merge(pd.read_parquet(f)[key + ["pred"]].rename(columns={"pred": n}), on=key, how="left")
models = [c for c in B.columns if c in ("persistence", "hist_mean", "seasonal_naive", "rolling_mean4") or "_" in c and c.split("_")[-1] in ("A", "B", "C", "D", "Bretro", "Cretro", "AR", "ARseas", "S")]
B["period"] = np.where(B.target_week >= "2023-01-01", "test", "tuning")
if outcome == "l_ed":
    B = B[B.target_week <= "2023-06-10"]          # ED field fully reported only to the week of 2023-06-04
B["regime"] = np.select([B.w4_retro == 1, (B.w4_retro == 0) & (B.w4 == 1)], ["smoke", "post_smoke"], "normal")
B["wk"] = B.target_week.dt.strftime("%Y%m%d")
B["east23"] = ((B.west == 0) & (B.target_week.between("2023-05-01", "2023-07-31"))).astype(int)
B["cap"] = np.where(B.base_beds >= B.drop_duplicates("hospital_pk").base_beds.median(), "high_capacity", "low_capacity")
E = {m: (B.target - B[m]).astype(float) for m in models}

def cl2(y, X, g1, g2):
    """OLS with two-way cluster-robust covariance (Cameron-Gelbach-Miller)."""
    ok = np.isfinite(y) & np.isfinite(X).all(1); y, X, g1, g2 = y[ok], X[ok], g1[ok], g2[ok]
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    def meat(g):
        S = pd.DataFrame(X * u[:, None]).groupby(g).sum().values
        return S.T @ S
    g12 = pd.factorize(pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str))[0]
    V = XtXi @ (meat(g1) + meat(g2) - meat(g12)) @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return b, se, len(y)

def dm(e_bench, e_model, mask):
    d = (e_bench ** 2 - e_model ** 2)[mask].values
    b, se, n = cl2(d, np.ones((len(d), 1)), B.state.values[mask], B.wk.values[mask])
    t = b[0] / se[0] if se[0] > 0 else np.nan
    return t, 1 - stats.norm.cdf(t)             # one-sided: model better than benchmark

rows = []
groups = {"all": np.ones(len(B), bool), "normal": B.regime == "normal", "smoke": B.regime == "smoke", "post_smoke": B.regime == "post_smoke",
          "E_low": B.E1_terc == 1, "E_mid": B.E1_terc == 2, "E_high": B.E1_terc == 3, "high_capacity": B.cap == "high_capacity", "low_capacity": B.cap == "low_capacity",
          "smoke_E_low": (B.regime == "smoke") & (B.E1_terc == 1), "smoke_E_high": (B.regime == "smoke") & (B.E1_terc == 3),
          "east_2023_episode": (B.east23 == 1) & (B.regime == "smoke"), "west": B.west == 1, "east": B.west == 0}
scale = (B.target - B.y_t).abs()
for per in ("tuning", "test"):
    for gname, g in groups.items():
        mk = (B.period == per) & np.asarray(g)
        if mk.sum() < 200: continue
        for m in models:
            e = E[m][mk].dropna()
            if len(e) < 200: continue
            mk2 = mk & E[m].notna() & E["persistence"].notna()
            ar = "ar_AR"
            r = dict(outcome=outcome, h=h, period=per, regime=gname, model=m, n=int(mk2.sum()), rmse=np.sqrt((e ** 2).mean()), mae=e.abs().mean(), mse=(e ** 2).mean(), bias=-e.mean(),
                     mase=e.abs().mean() / scale[mk2].mean(),
                     r2_vs_persistence=1 - (E[m][mk2] ** 2).mean() / (E["persistence"][mk2] ** 2).mean())
            if ar in E:
                mk3 = mk2 & E[ar].notna()
                r["r2_vs_ar"] = 1 - (E[m][mk3] ** 2).mean() / (E[ar][mk3] ** 2).mean()
                if m not in ("persistence", ar):
                    r["dm_t_vs_ar"], r["dm_p_vs_ar"] = dm(E[ar], E[m], mk3.values)
            if m != "persistence":
                r["dm_t_vs_persistence"], r["dm_p_vs_persistence"] = dm(E["persistence"], E[m], mk2.values)
            rows.append(r)
M = pd.DataFrame(rows); M.to_csv(f"{ML}/metrics_{TAG}.csv", index=False)

# incremental value of information sets (same learner, same observations): B vs A, C vs B, D vs C, retro vs real-time
inc = []
for per in ("tuning", "test"):
    for gname in ("all", "normal", "smoke", "post_smoke", "smoke_E_low", "smoke_E_high", "east_2023_episode"):
        for learner in sorted({m.rsplit("_", 1)[0] for m in models if "_" in m and m.split("_")[-1] in ("A", "B", "C", "D")}):
            for small, big in (("A", "B"), ("B", "C"), ("C", "D"), ("B", "Bretro"), ("A", "Bretro"), ("Bretro", "Cretro")):
                a, b_ = f"{learner}_{small}", f"{learner}_{big}"
                if a not in E or b_ not in E: continue
                mk = ((B.period == per) & np.asarray(groups[gname]) & E[a].notna() & E[b_].notna()).values
                if mk.sum() < 200: continue
                t, p = dm(E[a], E[b_], mk)
                r = dict(outcome=outcome, h=h, period=per, regime=gname, learner=learner, small=small, big=big, n=int(mk.sum()),
                         r2_big_vs_small=1 - (E[b_][mk] ** 2).mean() / (E[a][mk] ** 2).mean(), rmse_small=np.sqrt((E[a][mk] ** 2).mean()), rmse_big=np.sqrt((E[b_][mk] ** 2).mean()), dm_t=t, dm_p=p)
                if learner == "ols":            # Clark-West: valid for nested linear models estimated by least squares
                    f = (E[a] ** 2 - (E[b_] ** 2 - (B[a] - B[b_]).astype(float) ** 2))[mk].values
                    bb, se, _ = cl2(f, np.ones((len(f), 1)), B.state.values[mk], B.wk.values[mk]); r["cw_t"] = bb[0] / se[0]; r["cw_p"] = 1 - stats.norm.cdf(r["cw_t"])
                inc.append(r)
I = pd.DataFrame(inc); I.to_csv(f"{ML}/incremental_{TAG}.csv", index=False)

# H6 / H7: does the forecast loss, or the gain over persistence, differ by smoke regime and experience? (direct test of differences)
h67 = []
T = B[B.period == "test"]
for m in [x for x in ("lgbm_C", "lgbm_A", "ridge_C", "ar_AR", "persistence") if x in E]:
    e2 = (E[m] ** 2)[T.index]; gain = (E["persistence"] ** 2 - E[m] ** 2)[T.index]
    smoke = (T.regime == "smoke").astype(float).values; E1 = T.E1.astype(float).values
    X = np.column_stack([np.ones(len(T)), smoke, E1, smoke * E1])
    for lab, yv in (("squared_error", e2.values), ("gain_over_persistence", gain.values)):
        if m == "persistence" and lab != "squared_error": continue
        b, se, n = cl2(yv, X, T.state.values, T.wk.values)
        for j, nm in enumerate(("const", "smoke", "E1", "smoke_x_E1")):
            h67.append(dict(outcome=outcome, h=h, model=m, dep=lab, term=nm, coef=b[j], se=se[j], p=2 * stats.norm.sf(abs(b[j] / se[j])) if se[j] > 0 else np.nan, n=n))
pd.DataFrame(h67).to_csv(f"{ML}/h6_h7_tests_{TAG}.csv", index=False)

# week-block bootstrap CI for R2_OOS (test period)
rng = np.random.default_rng(SEED); boot = []
wks = T.wk.unique()
for m in [x for x in ("lgbm_C", "lgbm_B", "lgbm_A", "xgb_C", "ridge_C", "ols_C", "ar_AR") if x in E]:
    for gname in ("all", "smoke", "normal"):
        mk = np.asarray(groups[gname])[T.index] if False else (np.asarray(groups[gname]) & (B.period == "test"))
        d = pd.DataFrame({"wk": B.wk[mk], "a": (E["persistence"] ** 2)[mk], "b": (E[m] ** 2)[mk]}).dropna().groupby("wk").sum()
        if len(d) < 5: continue
        dr = [1 - s.b.sum() / s.a.sum() for s in (d.iloc[rng.integers(0, len(d), len(d))] for _ in range(1000))]
        boot.append(dict(outcome=outcome, h=h, model=m, regime=gname, r2_vs_persistence=1 - d.b.sum() / d.a.sum(), ci_lo=np.percentile(dr, 2.5), ci_hi=np.percentile(dr, 97.5), n_weeks=len(d)))
pd.DataFrame(boot).to_csv(f"{ML}/r2_bootstrap_{TAG}.csv", index=False)

# split-conformal 90% intervals: quantile of absolute errors in the tuning period, applied to the test period
iv = []
for m in [x for x in ("lgbm_C", "ridge_C", "ar_AR", "persistence") if x in E]:
    q = E[m][B.period == "tuning"].abs().quantile(0.90)
    for gname in ("all", "normal", "smoke", "post_smoke", "E_low", "E_high", "east_2023_episode"):
        mk = (B.period == "test") & np.asarray(groups[gname]) & E[m].notna()
        if mk.sum() < 100: continue
        iv.append(dict(outcome=outcome, h=h, model=m, regime=gname, nominal=0.90, coverage=(E[m][mk].abs() <= q).mean(), width=2 * q, n=int(mk.sum())))
pd.DataFrame(iv).to_csv(f"{ML}/intervals_{TAG}.csv", index=False)

# extreme-congestion detection: target above the hospital's own 90th percentile of the pre-2023 period
det = []
if outcome in ("occ", "icu_occ"):
    thr = B[B.period == "tuning"].groupby("hospital_pk").target.quantile(0.90).rename("thr"); Tt = B[B.period == "test"].merge(thr, on="hospital_pk")
    yv = (Tt.target >= Tt.thr).astype(int)
    for m in [x for x in models if x in Tt and x.split("_")[-1] in ("A", "B", "C", "D", "AR", "persistence") or x == "persistence"]:
        for gname, g in (("all", np.ones(len(Tt), bool)), ("smoke", Tt.regime == "smoke"), ("normal", Tt.regime == "normal")):
            ok = np.asarray(g) & Tt[m].notna().values
            if ok.sum() < 200 or yv[ok].nunique() < 2: continue
            s = (Tt[m] - Tt.thr)[ok]; pr, rc, f1, _ = precision_recall_fscore_support(yv[ok], (s >= 0).astype(int), average="binary", zero_division=0)
            det.append(dict(outcome=outcome, h=h, model=m, regime=gname, base_rate=yv[ok].mean(), auroc=roc_auc_score(yv[ok], s), pr_auc=average_precision_score(yv[ok], s), precision=pr, recall=rc, f1=f1, n=int(ok.sum())))
    # peak accuracy: RMSE in each hospital's top-decile target weeks
pd.DataFrame(det).to_csv(f"{ML}/detection_{TAG}.csv", index=False)
B[key + ["target_week", "target", "period", "regime", "E1_terc", "west"] + [m for m in ("persistence", "ar_AR", "ridge_C", "lgbm_A", "lgbm_B", "lgbm_C", "lgbm_Bretro") if m in B]].to_parquet(f"{ML}/eval_frame_{TAG}.parquet", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(M[(M.period == "test") & M.regime.isin(["all", "normal", "smoke"])].pivot_table(index="model", columns="regime", values=["rmse", "r2_vs_persistence", "r2_vs_ar"]).round(4).to_string())
print(I[(I.period == "test") & I.regime.isin(["all", "smoke"])].round(4).to_string())
print(pd.DataFrame(h67).round(4).to_string())
