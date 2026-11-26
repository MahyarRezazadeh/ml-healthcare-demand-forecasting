"""Context QC-D: daily walk-forward forecasts of ED stretcher occupancy (horizons 1, 3, 7 days). Exploratory (one smoke season)."""
import warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from config import *
P = pd.read_parquet(f"{PROC}/qc_panel.parquet"); P.loc[~P.occ.between(0, 300), "occ"] = np.nan
days = pd.date_range(P.date.min(), P.date.max()); G = pd.MultiIndex.from_product([P.ed_id.unique(), days], names=["ed_id", "date"]).to_frame(index=False).merge(P, on=["ed_id", "date"], how="left")
G = G.sort_values(["ed_id", "date"]).reset_index(drop=True); g = G.groupby("ed_id"); G["E"] = g.E.transform(lambda s: s.ffill().bfill()); G["region"] = g.region.transform(lambda s: s.ffill().bfill())
rows, tests = [], []
for h in (1, 3, 7):
    F = G[["ed_id", "date", "region"]].copy(); F["y_t"] = G.occ; F["target"] = g.occ.shift(-h); F["tdate"] = F.date + pd.Timedelta(days=h)
    for k in (1, 2, 3, 6, 7, 13): F[f"l{k}"] = g.occ.shift(k)
    F["y_same_dow"] = g.occ.shift(7 - h)                                   # last observed value on the target's weekday
    F["rm7"] = g.occ.transform(lambda s: s.rolling(7, min_periods=4).mean()); F["hmean"] = g.occ.transform(lambda s: s.expanding(min_periods=14).mean())
    for d_ in range(7): F[f"dow{d_}"] = (F.tdate.dt.dayofweek == d_).astype(float)
    F["stretchers"] = G.stretchers; F["doy_s"] = np.sin(2 * np.pi * F.tdate.dt.dayofyear / 365.25); F["doy_c"] = np.cos(2 * np.pi * F.tdate.dt.dayofyear / 365.25)
    for c in ("spm", "spm_lag1", "spm_lag2", "spm_lag3", "plume_heavy", "plume_any", "pm_total"): F[c] = G[c]
    F["spm_retro"] = g.spm.shift(-h); F["heavy_retro"] = g.plume_heavy.shift(-h); F["E"] = G.E; F["Exspm"] = F.E * F.spm; F["Exspm_retro"] = F.E * F.spm_retro
    F = F.dropna(subset=["target", "y_t", "l7"]).reset_index(drop=True); F["dy"] = F.target - F.y_t; F["ti"] = (F.date - days[0]).dt.days
    A = ["y_t", "l1", "l2", "l3", "l6", "l7", "l13", "y_same_dow", "rm7", "hmean", "stretchers", "doy_s", "doy_c"] + [f"dow{d_}" for d_ in range(7)]
    SETS = {"A": A, "B": A + ["spm", "spm_lag1", "spm_lag2", "spm_lag3", "plume_heavy", "plume_any", "pm_total"]}
    SETS["C"] = SETS["B"] + ["E", "Exspm"]; SETS["Bretro"] = SETS["B"] + ["spm_retro", "heavy_retro"]; SETS["Cretro"] = SETS["Bretro"] + ["E", "Exspm", "Exspm_retro"]
    lin = lambda e: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), e)
    MODELS = {"ols": lambda: lin(LinearRegression()), "ridge": lambda: lin(Ridge(alpha=10)),
              "lgbm": lambda: lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=3)}
    first = int((pd.Timestamp("2023-03-01") - days[0]).days); y = F.dy.values; ti = F.ti.values
    F["persistence"] = F.y_t; F["seasonal_naive"] = F.y_same_dow.fillna(F.y_t); F["hist_mean"] = F.hmean
    for name, mk in MODELS.items():
        for fs, cols in SETS.items():
            X = F[cols].values.astype(np.float32); pred = np.full(len(F), np.nan); m = None
            for t in range(first, ti.max() + 1):
                if m is None or (t - first) % 14 == 0:
                    tr = np.where(ti + h <= t)[0]; m = mk().fit(X[tr], y[tr])
                te = np.where(ti == t)[0]
                if len(te): pred[te] = m.predict(X[te])
            F[f"{name}_{fs}"] = F.y_t + pred
    ev = F[F.ti >= first].copy(); ev["smoke"] = (ev.spm_retro >= SEVERE_PM) | (ev.heavy_retro == 1); ev["d"] = ev.tdate.dt.strftime("%Y%m%d")
    ev["Ehigh"] = ev.E > ev.drop_duplicates("ed_id").E.median()
    mods = ["persistence", "seasonal_naive", "hist_mean"] + [f"{n}_{s}" for n in MODELS for s in SETS]
    def dmt(a, b, mk):            # mean loss differential, clustered by date
        d = ((ev.target - ev[a]) ** 2 - (ev.target - ev[b]) ** 2)[mk]; s = d.groupby(ev.d[mk]).sum(); n = len(s)
        se = np.sqrt(n / (n - 1) * (s ** 2).sum()) / len(d) if d.sum() != 0 else np.nan; t = d.mean() / se; return t, 1 - stats.norm.cdf(t)
    for reg, mk in (("all", ev.target.notna()), ("smoke", ev.smoke), ("normal", ~ev.smoke), ("smoke_E_high", ev.smoke & ev.Ehigh), ("smoke_E_low", ev.smoke & ~ev.Ehigh)):
        for m in mods:
            e = (ev.target - ev[m])[mk]; ep = (ev.target - ev.persistence)[mk]; es = (ev.target - ev.seasonal_naive)[mk]
            r = dict(context="QC-D", outcome="occ", h=h, regime=reg, model=m, n=int(mk.sum()), rmse=np.sqrt((e ** 2).mean()), mae=e.abs().mean(), bias=-e.mean(),
                     r2_vs_persistence=1 - (e ** 2).mean() / (ep ** 2).mean(), r2_vs_seasonal_naive=1 - (e ** 2).mean() / (es ** 2).mean())
            if m.endswith("_A") is False and "_" in m and m.split("_")[0] in MODELS:
                r["dm_t_vs_ols_A"], r["dm_p_vs_ols_A"] = dmt("ols_A", m, mk)
            rows.append(r)
        for n_ in MODELS:
            for small, big in (("A", "B"), ("B", "C"), ("A", "Bretro"), ("Bretro", "Cretro")):
                t, p = dmt(f"{n_}_{small}", f"{n_}_{big}", mk)
                tests.append(dict(context="QC-D", h=h, regime=reg, learner=n_, small=small, big=big, n=int(mk.sum()), n_dates=int(ev.d[mk].nunique()),
                                  r2_big_vs_small=1 - ((ev.target - ev[f"{n_}_{big}"])[mk] ** 2).mean() / ((ev.target - ev[f"{n_}_{small}"])[mk] ** 2).mean(), dm_t=t, dm_p=p))
    print("h", h, "done", flush=True)
    if h == 1: ev[["ed_id", "date", "tdate", "target", "smoke", "persistence", "seasonal_naive", "ols_A", "lgbm_A", "lgbm_B", "lgbm_C", "lgbm_Bretro"]].to_parquet(f"{ML}/qcd_eval_frame_h1.parquet", index=False)
R = pd.DataFrame(rows); R.to_csv(f"{ML}/qcd_metrics.csv", index=False); T = pd.DataFrame(tests); T.to_csv(f"{ML}/qcd_incremental.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R[R.regime.isin(["all", "smoke"])].pivot_table(index=["h", "model"], columns="regime", values=["rmse", "r2_vs_persistence", "r2_vs_seasonal_naive"]).round(3).to_string())
print(T[T.regime.isin(["all", "smoke"])].round(4).to_string())
