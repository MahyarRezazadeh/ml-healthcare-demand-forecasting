"""Expanding-window walk-forward forecasts, context US-W.
Usage: python 11_forecast_walkforward.py <outcome> <h> <tier>      tier = 1 (linear + boosted, all feature sets) | 2 (other learners, sets A/B/C)
Targets are modelled as the change y(t+h) - y(t); the level forecast is y(t) + predicted change, so persistence = zero change.
Checkpointed: one parquet of predictions per (model, feature set); finished files are skipped.
"""
import sys, os, time, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import lightgbm as lgb, xgboost as xgb, catboost as cb
from config import *
import features

outcome, h, tier = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cov = {"occ": "occ_cov", "icu_occ": "icu_cov", "l_ed": "ed_cov"}[outcome]
F = features.subsample(features.build(outcome, h, cov))
F["E1xw1r"] = F.E1 * F.w1_retro
F["dy"] = F.target - F.y_t
OUT = f"{ML}/preds_{outcome}_h{h}"; os.makedirs(OUT, exist_ok=True)
weeks = np.sort(F.week.unique()); T = {w: i for i, w in enumerate(np.sort(pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=["week"]).week.unique()))}
FIRST = 52                                    # first forecast origin index
TEST0 = T[pd.Timestamp("2023-01-01")]         # frozen-hyperparameter final test starts here
rng = np.random.default_rng(SEED)
HP = json.load(open(f"{ML}/tuned_hyperparameters.json")) if os.path.exists(f"{ML}/tuned_hyperparameters.json") else {}
hp = lambda name, default: {**default, **HP.get(name, {})}

def lin(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)

MODELS1 = {
    "ols": lambda: lin(LinearRegression()),
    "ridge": lambda: lin(Ridge(**hp("ridge", {"alpha": 10.0}))),
    "lasso": lambda: lin(Lasso(**hp("lasso", {"alpha": 0.01}), max_iter=2000, tol=1e-3)),
    "enet": lambda: lin(ElasticNet(**hp("enet", {"alpha": 0.01, "l1_ratio": 0.5}), max_iter=2000, tol=1e-3)),
    "lgbm": lambda: lgb.LGBMRegressor(**hp("lgbm", {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}), random_state=SEED, n_jobs=3, verbose=-1),
    "xgb": lambda: xgb.XGBRegressor(**hp("xgb", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}), tree_method="hist", random_state=SEED, n_jobs=3),
}
MODELS2 = {
    "tree": lambda: make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(**hp("tree", {"max_depth": 8, "min_samples_leaf": 200}), random_state=SEED)),
    "rf": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**hp("rf", {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 50, "max_features": 0.5}), max_samples=0.5, n_jobs=3, random_state=SEED)),
    "extratrees": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(**hp("extratrees", {"n_estimators": 100, "max_depth": 14, "min_samples_leaf": 50, "max_features": 0.5}), n_jobs=3, random_state=SEED)),
    "gboost": lambda: HistGradientBoostingRegressor(**hp("gboost", {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 100}), random_state=SEED),
    "catboost": lambda: cb.CatBoostRegressor(**hp("catboost", {"iterations": 300, "learning_rate": 0.08, "depth": 6}), random_seed=SEED, thread_count=3, verbose=0),
    "mlp": lambda: lin(MLPRegressor(**hp("mlp", {"hidden_layer_sizes": (64, 32), "alpha": 1e-3}), early_stopping=True, max_iter=60, random_state=SEED)),
    "svr": lambda: lin(SVR(**hp("svr", {"C": 1.0, "epsilon": 0.5, "gamma": "scale"}))),
}
MAXTRAIN = {"svr": 12000}                     # SVR is O(n^2): random subsample of the training window (logged)
TIER2_CAP = 100000                            # tier-2 learners train on a random subsample of at most this many rows

def walk(name, make, fs, refit_every, cap=None):
    f = f"{OUT}/{name}_{fs}.parquet"
    if os.path.exists(f):
        return
    X = F[features.SETS[fs]].values.astype(np.float32); y = F.dy.values; ti = F.t_idx.values
    pred = np.full(len(F), np.nan, np.float32); model = None; t0 = time.time()
    for t in range(FIRST, ti.max() + 1):
        if model is None or (t - FIRST) % refit_every == 0:
            tr = np.where(ti + h <= t)[0]                 # targets observed by origin t
            if cap and len(tr) > cap:
                tr = rng.choice(tr, cap, replace=False)
            model = make().fit(X[tr], y[tr])
        te = np.where(ti == t)[0]
        if len(te):
            pred[te] = model.predict(X[te])
    o = F.loc[F.t_idx >= FIRST, ["hospital_pk", "week", "target_week"]].copy(); o["pred"] = (F.y_t.values + pred)[F.t_idx.values >= FIRST]
    o.to_parquet(f, index=False)
    print(name, fs, f"{time.time() - t0:.0f}s", flush=True)

if tier == 1:
    # benchmarks (no fitting beyond expanding means)
    b = F.loc[F.t_idx >= FIRST, ["hospital_pk", "week", "target_week", "state", "division", "t_idx", "target", "y_t", "y_seas", "y_hmean", "y_rm4",
                                 "w4", "w4_retro", "w1_retro", "w2_retro", "w1", "E1", "E1_terc", "base_beds", "base_occ", "west", "fema_retro"]].copy()
    b["persistence"] = b.y_t; b["hist_mean"] = b.y_hmean; b["seasonal_naive"] = b.y_seas.fillna(b.y_hmean); b["rolling_mean4"] = b.y_rm4
    b.to_parquet(f"{OUT}/_benchmarks.parquet", index=False)
    # pooled AR(4): the strong conventional benchmark
    features.SETS["AR"] = ["y_t", "y_l1", "y_l2", "y_l3", "y_hmean"]
    features.SETS["ARseas"] = ["y_t", "y_l1", "y_l2", "y_l3", "y_hmean", "y_seas", "sin1", "cos1", "sin2", "cos2"]
    walk("ar", MODELS1["ols"], "AR", 4); walk("ar", MODELS1["ols"], "ARseas", 4)
    sets = ["A", "B", "C", "D", "Bretro", "Cretro"] if (outcome == "occ" and h == 1) else ["A", "B", "C", "Bretro"]
    for name, make in MODELS1.items():
        if not (outcome == "occ" and h == 1) and name not in ("ols", "ridge", "lgbm"): continue     # secondary targets: reduced learner set (computing limit)
        for fs in sets:
            if name == 'xgb' and fs not in ('A', 'B', 'C'): continue      # computing limit: XGBoost on sets A-C only
            walk(name, make, fs, 13 if name in ('lasso', 'enet') else 4)   # lasso/enet refit quarterly (coordinate descent is slow at the tuned penalty)
else:
    for name, make in MODELS2.items():
        for fs in ("A", "B", "C"):
            try:
                walk(name, make, fs, 26, cap=MAXTRAIN.get(name, TIER2_CAP))   # tier 2 refit every 26 weeks (computing limit)
            except Exception as ex:
                open(f"{ML}/model_failures.log", "a").write(f"{outcome} h{h} {name} {fs}: {repr(ex)[:300]}\n"); print("FAILED", name, fs, ex, flush=True)
print("done")
