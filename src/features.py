"""Feature frames for forecasting (protocol section 10). One row = (hospital, forecast origin t); target = y at t+h.
Everything in a real-time set is known at the end of week t. '_retro' columns use the target week and are never called forecasts.
"""
import os, numpy as np, pandas as pd, pyarrow.parquet as pq
from config import *

def build(outcome="occ", h=1, cov="occ_cov"):
    cache = f"{PROC}/ml_{outcome}_h{h}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    cols = ["hospital_pk", "week", "state", "division", "west", "hospital_subtype", "is_metro_micro", "lat", "lon", "occ", "icu_occ", "ed", "beds", "icu_beds",
            "covid_share", cov, "w1", "w2", "w3", "w4", "w5", "w1_lag1", "w1_lag2", "w2_lag1", "w1_cum4", "tmax", "prcp", "E1", "E1_terc", "yrs_since_last_sev",
            "E2", "E6", "med_income", "poverty", "share65", "places_copd", "pop", "fema_fire_pm4"]
    P = pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=list(dict.fromkeys(cols)))
    P = P[(P.hospital_subtype == "Short Term") & (P[cov] >= MIN_COVERAGE)].copy()
    P["l_ed"] = np.log1p(P.ed)
    # complete hospital x week grid so that shifts are true calendar shifts
    weeks = np.sort(P.week.unique()); hosp = P.hospital_pk.unique()
    G = pd.MultiIndex.from_product([hosp, weeks], names=["hospital_pk", "week"]).to_frame(index=False).merge(P, on=["hospital_pk", "week"], how="left")
    static = ["state", "division", "west", "is_metro_micro", "lat", "lon", "E1", "E1_terc", "yrs_since_last_sev", "E2", "E6", "med_income", "poverty", "share65", "places_copd", "pop"]
    G[static] = G.groupby("hospital_pk")[static].transform(lambda s: s.ffill().bfill())
    G = G.sort_values(["hospital_pk", "week"]).reset_index(drop=True)
    g = G.groupby("hospital_pk")
    y = outcome
    F = G[["hospital_pk", "week", "state", "division"]].copy()
    F["y_t"] = G[y]; F["target"] = g[y].shift(-h); F["target_week"] = g["week"].shift(-h)
    for k in (1, 2, 3, 7):
        F[f"y_l{k}"] = g[y].shift(k)
    F["y_rm4"] = g[y].transform(lambda s: s.rolling(4, min_periods=2).mean()); F["y_rm8"] = g[y].transform(lambda s: s.rolling(8, min_periods=3).mean())
    F["y_sd8"] = g[y].transform(lambda s: s.rolling(8, min_periods=3).std())
    F["y_seas"] = g[y].shift(52 - h)                       # same calendar week one year before the target
    F["y_hmean"] = g[y].transform(lambda s: s.expanding(min_periods=4).mean())
    woy = F.target_week.dt.isocalendar().week.astype(float)
    F["sin1"] = np.sin(2 * np.pi * woy / 52.18); F["cos1"] = np.cos(2 * np.pi * woy / 52.18); F["sin2"] = np.sin(4 * np.pi * woy / 52.18); F["cos2"] = np.cos(4 * np.pi * woy / 52.18)
    F["metro"] = (G.is_metro_micro.astype(str).str.lower() == "true").astype(float)
    F["lat"], F["lon"], F["west"] = G.lat, G.lon, G.west
    F["covid_share"] = G.covid_share; F["l_beds_t"] = np.log(G.beds.where(G.beds > 0))
    # B: environment known at t
    for c in ("w1", "w2", "w3", "w4", "w5", "w1_lag1", "w1_lag2", "w2_lag1", "w1_cum4", "tmax", "prcp"):
        F[c] = G[c]
    # retrospective: realised target-week environment
    for c in ("w1", "w2", "w3", "w4", "tmax", "prcp"):
        F[f"{c}_retro"] = g[c].shift(-h)
    F["fema_retro"] = g["fema_fire_pm4"].shift(-h)
    # C: experience
    F["E1"], F["E1_terc"], F["yrs_since"], F["E2"], F["E6"] = G.E1, G.E1_terc, G.yrs_since_last_sev, G.E2, G.E6
    F["E1xw1"] = F.E1 * F.w1; F["E1xw2"] = F.E1 * F.w2; F["E1xw3"] = F.E1 * F.w3
    # D: capacity and vulnerability (baseline = first 12 window weeks, before the first forecast origin)
    b = G[G.week <= weeks[11]].groupby("hospital_pk").agg(base_beds=("beds", "mean"), base_icu=("icu_beds", "mean"), base_occ=("occ", "mean"))
    F = F.merge(b, left_on="hospital_pk", right_index=True, how="left")
    F["base_beds"] = np.log1p(F.base_beds); F["base_icu"] = np.log1p(F.base_icu)
    for c in ("med_income", "poverty", "share65", "places_copd", "pop"):
        F[c] = G[c].values
    F["pop"] = np.log(F["pop"])
    F["t_idx"] = F.week.map({w: i for i, w in enumerate(weeks)})
    F = F.dropna(subset=["target", "y_t"])
    F = F[F.t_idx >= 8]
    for c in F.select_dtypes("float64").columns:
        F[c] = F[c].astype("float32")
    F.to_parquet(cache, index=False)
    return F

A = ["y_t", "y_l1", "y_l2", "y_l3", "y_l7", "y_rm4", "y_rm8", "y_sd8", "y_seas", "y_hmean", "sin1", "cos1", "sin2", "cos2", "metro", "lat", "lon", "west", "covid_share", "l_beds_t"]
B_ADD = ["w1", "w2", "w3", "w4", "w5", "w1_lag1", "w1_lag2", "w2_lag1", "w1_cum4", "tmax", "prcp"]
RETRO_ADD = ["w1_retro", "w2_retro", "w3_retro", "w4_retro", "tmax_retro", "prcp_retro"]
C_ADD = ["E1", "E1_terc", "yrs_since", "E2", "E6", "E1xw1", "E1xw2", "E1xw3"]
D_ADD = ["base_beds", "base_icu", "base_occ", "med_income", "poverty", "share65", "places_copd", "pop"]
SETS = {"A": A, "B": A + B_ADD, "C": A + B_ADD + C_ADD, "D": A + B_ADD + C_ADD + D_ADD,
        "Bretro": A + B_ADD + RETRO_ADD, "Cretro": A + B_ADD + RETRO_ADD + C_ADD + ["E1xw1r"]}


N_SUB = 1000
def subsample(F, n=N_SUB):
    """Fixed random sample of hospitals (seeded) - Amendment 2: full-sample forecasting exceeded the machine's memory."""
    h = np.sort(F.hospital_pk.unique())
    keep = np.random.default_rng(SEED).choice(h, min(n, len(h)), replace=False)
    return F[F.hospital_pk.isin(keep)].reset_index(drop=True)
