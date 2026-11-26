"""Hyperparameter tuning with an equal budget (8 random configurations per tunable learner).
Chronological split inside the pre-2023 period only: train on targets up to 2021-12-31, validate on origins in 2022.
Outcome occ, h=1, feature set C; the chosen values are then frozen for every outcome, horizon and feature set.
(Protocol asked for 20 configurations with walk-forward tuning; reduced to 8 on a single split for computing reasons - logged as a deviation.)
"""
import json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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
F = features.build("occ", 1, "occ_cov"); F["dy"] = F.target - F.y_t
X = F[features.SETS["C"]].values.astype(np.float32); y = F.dy.values
tr = np.where(F.target_week <= "2021-12-31")[0]; va = np.where((F.week >= "2022-01-01") & (F.target_week <= "2022-12-31"))[0]
rng = np.random.default_rng(SEED); trs = rng.choice(tr, min(len(tr), 120000), replace=False); trsv = rng.choice(tr, 10000, replace=False)
lin = lambda e: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), e)
imp = lambda e: make_pipeline(SimpleImputer(strategy="median"), e)
U = lambda a, b: float(rng.uniform(a, b)); LU = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b)))); I = lambda a, b: int(rng.integers(a, b + 1))
SPACE = {
 "ridge": (lambda: {"alpha": LU(0.01, 1000)}, lambda p: lin(Ridge(**p))),
 "lasso": (lambda: {"alpha": LU(1e-4, 1)}, lambda p: lin(Lasso(**p, max_iter=3000))),
 "enet": (lambda: {"alpha": LU(1e-4, 1), "l1_ratio": U(0.1, 0.9)}, lambda p: lin(ElasticNet(**p, max_iter=3000))),
 "lgbm": (lambda: {"n_estimators": I(150, 500), "learning_rate": LU(0.02, 0.15), "num_leaves": I(8, 64), "min_child_samples": I(30, 300), "subsample": U(0.6, 1), "subsample_freq": 1, "colsample_bytree": U(0.5, 1)}, lambda p: lgb.LGBMRegressor(**p, random_state=SEED, n_jobs=3, verbose=-1)),
 "xgb": (lambda: {"n_estimators": I(150, 500), "learning_rate": LU(0.02, 0.15), "max_depth": I(3, 8), "min_child_weight": I(10, 200), "subsample": U(0.6, 1), "colsample_bytree": U(0.5, 1)}, lambda p: xgb.XGBRegressor(**p, tree_method="hist", random_state=SEED, n_jobs=3)),
 "tree": (lambda: {"max_depth": I(3, 12), "min_samples_leaf": I(50, 1000)}, lambda p: imp(DecisionTreeRegressor(**p, random_state=SEED))),
 "rf": (lambda: {"n_estimators": 100, "max_depth": I(6, 16), "min_samples_leaf": I(20, 200), "max_features": U(0.3, 0.8)}, lambda p: imp(RandomForestRegressor(**p, max_samples=0.5, n_jobs=3, random_state=SEED))),
 "extratrees": (lambda: {"n_estimators": 100, "max_depth": I(6, 18), "min_samples_leaf": I(20, 200), "max_features": U(0.3, 0.8)}, lambda p: imp(ExtraTreesRegressor(**p, n_jobs=3, random_state=SEED))),
 "gboost": (lambda: {"max_iter": I(150, 500), "learning_rate": LU(0.02, 0.15), "max_leaf_nodes": I(8, 64), "min_samples_leaf": I(30, 300)}, lambda p: HistGradientBoostingRegressor(**p, random_state=SEED)),
 "catboost": (lambda: {"iterations": I(150, 400), "learning_rate": LU(0.03, 0.2), "depth": I(4, 8)}, lambda p: cb.CatBoostRegressor(**p, random_seed=SEED, thread_count=3, verbose=0)),
 "mlp": (lambda: {"hidden_layer_sizes": [int(v) for v in ((32,), (64, 32), (128, 64), (64, 64, 32))[I(0, 3)]], "alpha": LU(1e-5, 1e-1)}, lambda p: lin(MLPRegressor(**{**p, "hidden_layer_sizes": tuple(p["hidden_layer_sizes"])}, early_stopping=True, max_iter=60, random_state=SEED))),
 "svr": (lambda: {"C": LU(0.1, 30), "epsilon": LU(0.05, 2), "gamma": "scale"}, lambda p: lin(SVR(**p))),
}
import os
best = json.load(open(f"{ML}/tuned_hyperparameters.json")) if os.path.exists(f"{ML}/tuned_hyperparameters.json") else {}
log = pd.read_csv(f"{ML}/tuning_log.csv").to_dict("records") if os.path.exists(f"{ML}/tuning_log.csv") else []
base = float(np.mean(y[va] ** 2))
for name, (draw, make) in SPACE.items():
    if name in best: continue
    t0 = time.time(); bm = np.inf
    for k in range(8):
        p = draw()
        try:
            m = make(p).fit(X[trsv if name == "svr" else trs], y[trsv if name == "svr" else trs])
            mse = float(np.mean((y[va] - m.predict(X[va])) ** 2))
        except Exception as ex:
            mse = np.nan
        log.append(dict(model=name, config=json.dumps(p), val_mse=mse, val_r2_vs_persistence=1 - mse / base))
        if mse < bm:
            bm, best[name] = mse, p
    json.dump(best, open(f"{ML}/tuned_hyperparameters.json", "w"), indent=1); pd.DataFrame(log).to_csv(f"{ML}/tuning_log.csv", index=False)
    print(name, f"best R2 vs persistence {1 - bm / base:.4f}", f"{time.time() - t0:.0f}s", flush=True)
json.dump(best, open(f"{ML}/tuned_hyperparameters.json", "w"), indent=1)
pd.DataFrame(log).to_csv(f"{ML}/tuning_log.csv", index=False)
