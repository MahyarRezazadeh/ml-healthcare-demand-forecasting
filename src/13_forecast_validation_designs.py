"""Additional validation designs for US-W occupancy forecasts (h=1). Each part writes its own CSV and is skipped if present.
Usage: python 13_forecast_validation_designs.py [part ...]   parts: rolling lodo lho equal episode stack ts neural poisson
"""
import sys, os, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from config import *
import features
parts = sys.argv[1:] or ["rolling", "lodo", "lho", "equal", "episode", "stack", "ts", "neural", "poisson"]
F = features.subsample(features.build("occ", 1, "occ_cov")); F["dy"] = F.target - F.y_t; F = F.reset_index(drop=True)
HP = json.load(open(f"{ML}/tuned_hyperparameters.json"))
LG = lambda: lgb.LGBMRegressor(**HP["lgbm"], random_state=SEED, n_jobs=3, verbose=-1)
RD = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(**HP["ridge"]))
TEST = (F.target_week >= "2023-01-01").values; PRE = (F.target_week <= "2022-12-31").values
smoke = (F.w4_retro == 1).values
r2 = lambda e, mk: 1 - np.nanmean((F.dy.values[mk] - e[mk]) ** 2) / np.nanmean(F.dy.values[mk] ** 2)   # vs persistence (zero change)
rmse = lambda e, mk: float(np.sqrt(np.nanmean((F.dy.values[mk] - e[mk]) ** 2)))
done = lambda n: os.path.exists(f"{ML}/extra_{n}.csv")
X = {k: F[v].values.astype(np.float32) for k, v in features.SETS.items() if k in ("A", "B", "C", "Bretro")}
y = F.dy.values; ti = F.t_idx.values
rng = np.random.default_rng(SEED)

if "rolling" in parts and not done("rolling"):
    out = []
    for name, mk_, fs in (("lgbm", LG, "C"), ("ridge", RD, "C"), ("lgbm", LG, "A")):
        pred = np.full(len(F), np.nan); m = None
        for t in range(104 + 8, ti.max() + 1):
            if m is None or (t - 112) % 4 == 0:
                tr = np.where((ti + 1 <= t) & (ti + 1 > t - 104))[0]; m = mk_().fit(X[fs][tr], y[tr])
            te = np.where(ti == t)[0]
            if len(te): pred[te] = m.predict(X[fs][te])
        for g, mk in (("test_all", TEST), ("test_smoke", TEST & smoke), ("test_normal", TEST & ~smoke)):
            out.append(dict(design="rolling_104w", model=f"{name}_{fs}", regime=g, n=int(mk.sum()), rmse=rmse(pred, mk), r2_vs_persistence=r2(pred, mk)))
    pd.DataFrame(out).to_csv(f"{ML}/extra_rolling.csv", index=False); print(pd.DataFrame(out).round(4).to_string(), flush=True)

if "lodo" in parts and not done("lodo"):
    out = []
    for dv in sorted(F.division.unique()):
        ho = (F.division == dv).values
        for fs in ("A", "C"):
            m = LG().fit(X[fs][PRE & ~ho], y[PRE & ~ho]); pred = np.full(len(F), np.nan); pred[TEST & ho] = m.predict(X[fs][TEST & ho])
            m2 = LG().fit(X[fs][PRE], y[PRE]); pin = np.full(len(F), np.nan); pin[TEST & ho] = m2.predict(X[fs][TEST & ho])
            for g, mk in (("all", TEST & ho), ("smoke", TEST & ho & smoke)):
                if mk.sum() > 100:
                    out.append(dict(design="leave_one_division_out", held_out=dv, features=fs, regime=g, n=int(mk.sum()), r2_heldout=r2(pred, mk), r2_trained_with_division=r2(pin, mk)))
    pd.DataFrame(out).to_csv(f"{ML}/extra_lodo.csv", index=False); print(pd.DataFrame(out).round(4).to_string(), flush=True)

if "lho" in parts and not done("lho"):
    out = []; pred = {fs: np.full(len(F), np.nan) for fs in ("A", "C")}
    for tr, te in GroupKFold(5).split(X["A"], groups=F.hospital_pk.values):
        trm = np.zeros(len(F), bool); trm[tr] = True; tem = np.zeros(len(F), bool); tem[te] = True
        for fs in ("A", "C"):
            pred[fs][tem & TEST] = LG().fit(X[fs][trm & PRE], y[trm & PRE]).predict(X[fs][tem & TEST])
    for fs in ("A", "C"):
        for g, mk in (("all", TEST), ("smoke", TEST & smoke), ("normal", TEST & ~smoke)):
            out.append(dict(design="leave_hospital_out_5fold", features=fs, regime=g, n=int(mk.sum()), rmse=rmse(pred[fs], mk), r2_vs_persistence=r2(pred[fs], mk)))
    pd.DataFrame(out).to_csv(f"{ML}/extra_lho.csv", index=False); print(pd.DataFrame(out).round(4).to_string(), flush=True)

if "equal" in parts and not done("equal"):
    # algorithmic-adaptation check: is a group better forecast in smoke weeks only because it supplies more smoke-week training rows?
    out = []; lo = (F.E1_terc == 1).values; hi = (F.E1_terc == 3).values
    for rep in range(5):
        r = np.random.default_rng(SEED + rep)
        n_s = min((PRE & lo & smoke).sum(), (PRE & hi & smoke).sum()); n_n = min((PRE & lo & ~smoke).sum(), (PRE & hi & ~smoke).sum(), 60000)
        for gname, g in (("E_low", lo), ("E_high", hi)):
            eq = np.r_[r.choice(np.where(PRE & g & smoke)[0], n_s, replace=False), r.choice(np.where(PRE & g & ~smoke)[0], n_n, replace=False)]
            full = np.where(PRE & g)[0]
            for fs in ("A", "B", "Bretro"):
                for tname, tr in (("equal_sized", eq), ("own_full", full)):
                    p = np.full(len(F), np.nan); mk = TEST & g; p[mk] = LG().fit(X[fs][tr], y[tr]).predict(X[fs][mk])
                    for reg, mm in (("smoke", mk & smoke), ("normal", mk & ~smoke)):
                        out.append(dict(rep=rep, group=gname, training=tname, features=fs, regime=reg, n_train=len(tr), n_train_smoke=int(smoke[tr].sum()), n_test=int(mm.sum()), rmse=rmse(p, mm), r2_vs_persistence=r2(p, mm)))
            if rep: continue
    O = pd.DataFrame(out); O.to_csv(f"{ML}/extra_equal.csv", index=False)
    print(O.groupby(["training", "features", "regime", "group"])[["n_train", "n_train_smoke", "rmse", "r2_vs_persistence"]].mean().round(4).to_string(), flush=True)

if "episode" in parts and not done("episode"):
    out = []
    for q in ("2020Q4", "2021Q3", "2022Q3", "2023Q2", "2023Q3"):
        ho = (F.target_week.dt.to_period("Q").astype(str) == q).values
        for fs in ("A", "B", "Bretro"):
            m = LG().fit(X[fs][~ho], y[~ho]); p = np.full(len(F), np.nan); p[ho] = m.predict(X[fs][ho])
            for reg, mk in (("smoke", ho & smoke), ("all", ho)):
                if mk.sum() > 100:
                    out.append(dict(design="leave_one_smoke_quarter_out (retrospective: trained on earlier AND later quarters)", quarter=q, features=fs, regime=reg, n=int(mk.sum()), rmse=rmse(p, mk), r2_vs_persistence=r2(p, mk)))
    pd.DataFrame(out).to_csv(f"{ML}/extra_episode.csv", index=False); print(pd.DataFrame(out).round(4).to_string(), flush=True)

if "stack" in parts and not os.path.exists(f"{ML}/preds_occ_h1/stack_C.parquet"):
    D = f"{ML}/preds_occ_h1"; base = pd.read_parquet(f"{D}/_benchmarks.parquet")[["hospital_pk", "week", "target_week", "target", "y_t"]]
    ms = [m for m in ("ols", "ridge", "lasso", "enet", "lgbm", "xgb") if os.path.exists(f"{D}/{m}_C.parquet")]
    for m in ms: base = base.merge(pd.read_parquet(f"{D}/{m}_C.parquet")[["hospital_pk", "week", "pred"]].rename(columns={"pred": m}), on=["hospital_pk", "week"])
    tr = (base.target_week <= "2022-12-31") & base[ms].notna().all(1); te = base.target_week >= "2023-01-01"
    st = Ridge(alpha=1.0, positive=True).fit(base.loc[tr, ms].values - base.loc[tr, ["y_t"]].values, (base.target - base.y_t)[tr])
    base["pred"] = np.nan; base.loc[te, "pred"] = base.loc[te, "y_t"] + st.predict(base.loc[te, ms].fillna(0).values - base.loc[te, ["y_t"]].values)
    base[["hospital_pk", "week", "target_week", "pred"]].to_parquet(f"{D}/stack_C.parquet", index=False)
    pd.Series(dict(zip(ms, st.coef_))).to_csv(f"{ML}/extra_stack_weights.csv"); print("stack weights", dict(zip(ms, st.coef_.round(3))), flush=True)

if "ts" in parts and not done("ts"):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    import logging; logging.getLogger("prophet").setLevel(logging.ERROR); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    full = F.groupby("hospital_pk").size(); hs = rng.choice(full[full >= 170].index.values, 150, replace=False)   # 150 hospitals from the 1,000-hospital sample
    rows = []; t0 = time.time()
    for i, hpk in enumerate(hs):
        d = F[F.hospital_pk == hpk].sort_values("t_idx"); yy = d.y_t.values; tw = d.week.values; n = len(d); te0 = int(np.searchsorted(d.target_week.values, np.datetime64("2023-01-01")))
        if te0 < 100 or n - te0 < 20: continue
        res = {k: np.full(n, np.nan) for k in ("arima", "sarimax_smoke", "prophet")}
        for a in range(te0, n, 13):
            b = min(a + 13, n)
            try:   # ARIMA(1,0,1)+const and with last week's smoke as exogenous regressor; parameters fixed between refits, state updated weekly
                for k, ex in (("arima", None), ("sarimax_smoke", np.nan_to_num(d.w1.values)[:, None])):
                    exl = None if ex is None else np.r_[[[0.0]], ex]              # length n+1: position j holds smoke observed in week j-1
                    m = SARIMAX(yy[:a + 1], exog=None if ex is None else exl[:a + 1], order=(1, 0, 1), trend="c").fit(disp=False, maxiter=50)
                    mm = m.append(yy[a + 1:b], exog=None if ex is None else exl[a + 1:b], refit=False) if b > a + 1 else m
                    pr = np.asarray(mm.predict(start=a + 1, end=b, exog=None if ex is None else exl[b:b + 1]))
                    # one-step-ahead predictions for positions a+1..b are the forecasts made at origins a..b-1
                    res[k][a:b] = pr
            except Exception as ex_:
                pass
            try:
                pm = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(pd.DataFrame({"ds": tw[:a + 1], "y": yy[:a + 1]}))
                fut = pd.DataFrame({"ds": d.target_week.values[a:b]}); res["prophet"][a:b] = pm.predict(fut).yhat.values
            except Exception as ex_:
                pass
        o = pd.DataFrame({"hospital_pk": hpk, "week": d.week.values, "target": d.target.values, "y_t": yy, "w4_retro": d.w4_retro.values, **res}).iloc[te0:]
        rows.append(o)
        if i % 20 == 0: print("ts", i, f"{time.time() - t0:.0f}s", flush=True)
    O = pd.concat(rows); O.to_parquet(f"{ML}/extra_ts_preds.parquet", index=False)
    # Prophet-XGBoost hybrid: XGBoost on real-time set B plus the Prophet forecast, trained on the first half of the test window, scored on the second half
    O = O.merge(F[["hospital_pk", "week"] + features.SETS["B"]].drop(columns=["y_t"]), on=["hospital_pk", "week"]); half = O.week.quantile(0.5)
    tr = (O.week <= half) & O.prophet.notna(); te = (O.week > half) & O.prophet.notna(); feats = [c for c in features.SETS["B"] if c != "y_t"] + ["y_t", "prophet"]
    hy = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=SEED, n_jobs=3).fit(O.loc[tr, feats], O.loc[tr, "target"]); O.loc[te, "prophet_xgb"] = hy.predict(O.loc[te, feats])
    ev = []
    for m in ("arima", "sarimax_smoke", "prophet", "prophet_xgb"):
        for reg, mk in (("all", O[m].notna()), ("smoke", O[m].notna() & (O.w4_retro == 1)), ("second_half", O[m].notna() & (O.week > half))):
            if mk.sum() > 50:
                ev.append(dict(model=m, regime=reg, n=int(mk.sum()), n_hospitals=O[mk].hospital_pk.nunique(), rmse=np.sqrt(((O.target - O[m])[mk] ** 2).mean()), rmse_persistence=np.sqrt(((O.target - O.y_t)[mk] ** 2).mean()),
                               r2_vs_persistence=1 - ((O.target - O[m])[mk] ** 2).mean() / ((O.target - O.y_t)[mk] ** 2).mean()))
    pd.DataFrame(ev).to_csv(f"{ML}/extra_ts.csv", index=False); print(pd.DataFrame(ev).round(4).to_string(), flush=True)

if "neural" in parts and not done("neural"):
    import torch, torch.nn as nn
    torch.manual_seed(SEED); torch.set_num_threads(3); L = 12
    P = pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=["hospital_pk", "week", "occ", "w1", "w2", "covid_share"])
    P = P[P.hospital_pk.isin(F.hospital_pk.unique())]; wk = np.sort(P.week.unique()); wi = {w: i for i, w in enumerate(wk)}; hi_ = {h_: i for i, h_ in enumerate(F.hospital_pk.unique())}
    A = np.full((len(hi_), len(wk), 4), np.nan, np.float32); A[P.hospital_pk.map(hi_).values, P.week.map(wi).values] = P[["occ", "w1", "w2", "covid_share"]].values
    hm = F.y_hmean.values; hh = F.hospital_pk.map(hi_).values; tt = F.t_idx.values
    seq = np.stack([A[hh, tt - L + 1 + k] for k in range(L)], 1)          # (N, L, 4): weeks t-L+1..t, all known at the origin
    seq[:, :, 0] -= hm[:, None]; ok = ~np.isnan(seq[:, :, 0]).any(1); seq = np.nan_to_num(seq)
    static = np.nan_to_num(np.column_stack([F.E1.values, F.sin1.values, F.cos1.values, F.metro.values]).astype(np.float32))
    trm = PRE & ok & (F.target_week <= "2022-06-30").values; vam = PRE & ok & (F.target_week > "2022-06-30").values; tem = TEST & ok
    mu, sd = seq[trm].mean((0, 1)), seq[trm].std((0, 1)) + 1e-6; seq = (seq - mu) / sd; ys = y.std()
    class RNN(nn.Module):
        def __init__(s, kind):
            super().__init__(); s.kind = kind
            s.core = nn.LSTM(4, 32, batch_first=True) if kind == "lstm" else nn.GRU(4, 32, batch_first=True) if kind == "gru" else nn.Sequential(
                nn.Conv1d(4, 32, 3, padding=2, dilation=1), nn.ReLU(), nn.Conv1d(32, 32, 3, padding=4, dilation=2), nn.ReLU(), nn.Conv1d(32, 32, 3, padding=8, dilation=4), nn.ReLU())
            s.head = nn.Sequential(nn.Linear(36, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, x, z):
            hcore = s.core(x.transpose(1, 2))[:, :, :x.shape[1]][:, :, -1] if s.kind == "tcn" else s.core(x)[0][:, -1]      # TCN: causal (left-padded, right-trimmed)
            return s.head(torch.cat([hcore, z], 1)).squeeze(1)
    T_ = lambda a: torch.tensor(a, dtype=torch.float32); out = []
    for kind in ("lstm", "gru", "tcn"):
        try:
            net = RNN(kind); opt = torch.optim.Adam(net.parameters(), 1e-3); idx = np.where(trm)[0]; best, bs = np.inf, None
            for ep in range(8):
                net.train(); r = np.random.default_rng(SEED + ep).permutation(idx)
                for k in range(0, len(r), 1024):
                    bi = r[k:k + 1024]; opt.zero_grad(); loss = ((net(T_(seq[bi]), T_(static[bi])) - T_(y[bi] / ys)) ** 2).mean(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): v = float(((net(T_(seq[vam]), T_(static[vam])) * ys - T_(y[vam])) ** 2).mean())
                if v < best: best, bs = v, {k_: v_.clone() for k_, v_ in net.state_dict().items()}
            net.load_state_dict(bs); net.eval(); p = np.full(len(F), np.nan)
            with torch.no_grad(): p[tem] = net(T_(seq[tem]), T_(static[tem])).numpy() * ys
            for reg, mk in (("test_all", tem), ("test_smoke", tem & smoke), ("test_normal", tem & ~smoke)):
                out.append(dict(model=kind, design="single chronological split (train to 2022-06, early stopping 2022-H2, test 2023-01 onward)", regime=reg, n=int(mk.sum()), rmse=rmse(p, mk), r2_vs_persistence=r2(p, mk)))
            print(kind, "done", flush=True)
        except Exception as ex:
            out.append(dict(model=kind, design="FAILED", regime=repr(ex)[:200])); print(kind, "FAILED", ex, flush=True)
    pd.DataFrame(out).to_csv(f"{ML}/extra_neural.csv", index=False); print(pd.DataFrame(out).round(4).to_string(), flush=True)

if "poisson" in parts and not done("poisson"):
    G = features.subsample(features.build("l_ed", 1, "ed_cov")); G = G[G.target_week <= "2023-06-10"]
    cnt = np.expm1(G.target.values); tr = (G.target_week <= "2022-12-31").values; te = ~tr; out = []
    for fs in ("A", "B", "C"):
        Xg = G[features.SETS[fs]].values
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=300)).fit(Xg[tr], cnt[tr]); mu_ = m.predict(Xg[te])
        dev = lambda mu__: 2 * np.mean(np.where(cnt[te] > 0, cnt[te] * np.log(cnt[te] / mu__), 0) - (cnt[te] - mu__))
        pers = np.maximum(np.expm1(G.y_t.values[te]), 0.5)
        out.append(dict(model=f"poisson_{fs}", n=int(te.sum()), poisson_deviance=dev(np.maximum(mu_, 0.5)), poisson_deviance_persistence=dev(pers), rmse_log=np.sqrt(np.mean((G.target.values[te] - np.log1p(mu_)) ** 2)),
                        rmse_log_persistence=np.sqrt(np.mean((G.target.values[te] - G.y_t.values[te]) ** 2))))
    pd.DataFrame(out).to_csv(f"{ML}/extra_poisson.csv", index=False); print(pd.DataFrame(out).round(4).to_string(), flush=True)
print("extra done")
