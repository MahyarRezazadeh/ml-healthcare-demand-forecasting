"""Double ML (partially linear) and causal forest for the effect of weekly smoke PM2.5 on inpatient occupancy, context US-W.
Target estimand: average partial effect of a 10 ug/m3 increase in weekly mean smoke PM2.5 on the occupancy rate (pp), conditional on
hospital FE and division x week FE (removed by within-transformation first) and flexible weather / location controls.
Identification is the same selection-on-fixed-effects-and-weather assumption as the panel model; ML adds flexibility, not exogeneity.
"""
import warnings, numpy as np, pandas as pd, pyfixest as pf, lightgbm as lgb
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from config import *
cols = ["hospital_pk", "week", "state", "division", "hospital_subtype", "is_metro_micro", "occ", "occ_cov", "beds", "w1", "w1_lag1", "w1_lag2", "tmax", "tmin", "prcp", "tmax_max",
        "E1", "E1_terc", "lat", "lon", "med_income", "share65", "places_copd", "fema_fire_pm4", "sample", "west"]
P = pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=cols)
P = P[(P.hospital_subtype == "Short Term") & (P.fema_fire_pm4 == 0) & (P.occ_cov >= MIN_COVERAGE)].dropna(subset=["occ", "w1", "tmax", "prcp", "E1"]).copy()
P["w"] = P.w1 / 10; P["divweek"] = P.division + P.week.dt.strftime("%Y%m%d"); P["metro"] = (P.is_metro_micro.astype(str).str.lower() == "true").astype(float)
b = P[P.week <= "2020-10-24"].groupby("hospital_pk").agg(base_occ=("occ", "mean"), base_beds=("beds", "mean")); P = P.merge(b, left_on="hospital_pk", right_index=True, how="left")
P["l_base_beds"] = np.log1p(P.base_beds); P["woy"] = P.week.dt.isocalendar().week.astype(float)
# within-transform y, w and time-varying controls on hospital + division-week FE
tv = ["occ", "w", "tmax", "tmin", "prcp", "tmax_max"]
dm = pf.feols("occ + w + tmax + tmin + prcp + tmax_max ~ 1 | hospital_pk + divweek", data=P) if False else None
from pyfixest.core.demean import demean
fe = np.column_stack([pd.factorize(P.hospital_pk)[0], pd.factorize(P.divweek)[0]]).astype(np.uint64)
R, ok = demean(P[tv].values.astype(float), fe, np.ones(len(P)), tol=1e-8)
for j, c in enumerate(tv): P[c + "_r"] = R[:, j]
P["tmax2_r"] = P.tmax_r ** 2; P["tp_r"] = P.tmax_r * P.prcp_r
Xc = ["tmax_r", "tmin_r", "prcp_r", "tmax_max_r", "tmax2_r", "tp_r", "tmax", "lat", "lon", "woy", "E1"]
rows = []

def dml_plr(d, label):
    y, w, X, g = d.occ_r.values, d.w_r.values, d[Xc].values, d.state.values
    ry, rw = np.zeros(len(d)), np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, groups=g):          # cross-fitting with whole states held out
        mk = dict(n_estimators=100, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, verbose=-1, random_state=SEED, n_jobs=3)
        ry[te] = y[te] - lgb.LGBMRegressor(**mk).fit(X[tr], y[tr]).predict(X[te])
        rw[te] = w[te] - lgb.LGBMRegressor(**mk).fit(X[tr], w[tr]).predict(X[te])
    th = (rw @ ry) / (rw @ rw); psi = rw * (ry - th * rw)
    S = pd.Series(psi).groupby(g).sum().values; G = len(S)
    se = np.sqrt(G / (G - 1) * (S ** 2).sum()) / (rw @ rw)    # state-clustered
    r2w = 1 - (rw ** 2).mean() / (w ** 2).mean()
    rows.append(dict(method="DML-PLR (LightGBM nuisances, state-grouped 5-fold cross-fitting)", sample=label, coef=th, se=se, p=2 * (1 - __import__("scipy").stats.norm.cdf(abs(th / se))), n=len(d), n_states=G,
                     share_w_var_explained_by_controls=r2w, resid_w_sd=rw.std()))
    return ry, rw

for lab, d in (("pooled", P), ("discovery", P[P["sample"] == "discovery"]), ("confirmation", P[P["sample"] == "confirmation"]), ("pooled_east", P[P.west == 0]), ("pooled_west", P[P.west == 1])):
    dml_plr(d, lab)
ry, rw = dml_plr(P, "pooled_for_heterogeneity"); rows.pop()
pd.DataFrame(rows).to_csv(f"{ECON}/usw_dml_results.csv", index=False); print(pd.DataFrame(rows).round(4).to_string())

# overlap / support of the continuous treatment after partialling out
ov = pd.DataFrame({"E1_terc": P.E1_terc.values, "rw": rw, "w": P.w.values}).groupby("E1_terc").agg(n=("rw", "size"), resid_w_sd=("rw", "std"), share_w_positive=("w", lambda s: (s > 0).mean()),
        share_w_ge1=("w", lambda s: (s >= 1).mean()), n_w_ge2=("w", lambda s: int((s >= 2).sum())))
ov.to_csv(f"{ECON}/usw_dml_overlap_by_experience.csv"); print(ov.round(4).to_string())

# heterogeneity: R-learner best linear projection of the effect on predetermined characteristics (state-clustered)
Z = P[["E1", "l_base_beds", "base_occ", "metro", "med_income", "share65", "west"]].copy()
Z = (Z - Z.mean()) / Z.std(); Z.insert(0, "const", 1.0); okz = Z.notna().all(1).values
A = Z.values[okz] * rw[okz, None]; bb = np.linalg.lstsq(A, ry[okz], rcond=None)[0]; u = ry[okz] - A @ bb
S = pd.DataFrame(A * u[:, None]).groupby(P.state.values[okz]).sum().values; bread = np.linalg.inv(A.T @ A); V = bread @ (S.T @ S) @ bread
blp = pd.DataFrame({"term": Z.columns, "coef": bb, "se": np.sqrt(np.diag(V))}); blp["p"] = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(blp.coef / blp.se)))
blp.to_csv(f"{ECON}/usw_dml_heterogeneity_blp.csv", index=False); print(blp.round(4).to_string())

# causal forest (econml) on a random subsample of hospitals for memory; honest, grouped by state
try:
    from econml.dml import CausalForestDML
    rng = np.random.default_rng(SEED); hs = rng.choice(P.hospital_pk.unique(), 500, replace=False); d = P[P.hospital_pk.isin(hs) & okz]
    Xh = d[["E1", "l_base_beds", "base_occ", "metro", "med_income", "share65", "lat", "lon"]].values
    cf = CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=100, verbose=-1, n_jobs=3), model_t=lgb.LGBMRegressor(n_estimators=100, verbose=-1, n_jobs=3), n_estimators=200, min_samples_leaf=200,
                         max_samples=0.3, cv=GroupKFold(4), random_state=SEED, n_jobs=3)
    cf.fit(d.occ_r.values, d.w_r.values, X=Xh, W=d[["tmax_r", "prcp_r", "tmax2_r", "woy"]].values, groups=d.state.values)
    d = d.assign(cate=cf.effect(Xh))
    out = d.groupby("E1_terc").cate.agg(["mean", "std", "size"]); out.loc["ATE"] = [cf.ate(Xh), np.nan, len(d)]
    out.to_csv(f"{ECON}/usw_causal_forest_cate_by_experience.csv")
    imp = pd.Series(cf.feature_importances_, index=["E1", "l_base_beds", "base_occ", "metro", "med_income", "share65", "lat", "lon"]).sort_values(ascending=False)
    imp.to_csv(f"{ECON}/usw_causal_forest_split_importance.csv"); print(out.round(4).to_string()); print(imp.round(3).to_string())
    d.groupby("hospital_pk").agg(cate=("cate", "mean"), E1=("E1", "first"), west=("west", "first")).to_csv(f"{ECON}/usw_causal_forest_hospital_cate.csv")
except Exception as ex:
    open(f"{ECON}/usw_causal_forest_FAILED.txt", "w").write(repr(ex)); print("causal forest failed:", ex)
