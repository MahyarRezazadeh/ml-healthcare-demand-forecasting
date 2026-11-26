"""Descriptive tables and all figures (PDF + 300-dpi PNG). Each figure is wrapped so that a missing input skips only that figure."""
import glob, traceback, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.pdf"); fig.savefig(f"{FIG}/{name}.png", dpi=300); plt.close(fig); print("saved", name)
def attempt(fn):
    try: fn()
    except Exception: print("SKIPPED", fn.__name__, traceback.format_exc().splitlines()[-1])

def desc():
    cols = ["hospital_pk", "week", "west", "hospital_subtype", "occ", "icu_occ", "ed", "beds", "w1", "w2", "w3", "w4", "sev_days", "pm_total", "E1_raw", "sev_days_0619", "E1_terc", "sample", "tmax", "occ_cov"]
    P = pd.read_parquet(f"{PROC}/analysis_panel.parquet", columns=cols); P = P[P.hospital_subtype == "Short Term"]
    v = ["occ", "icu_occ", "ed", "beds", "w1", "w2", "w3", "w4", "sev_days", "pm_total", "sev_days_0619", "tmax"]
    S = pd.concat({k: g[v].describe().T[["count", "mean", "std", "min", "50%", "max"]] for k, g in [("all", P), ("discovery", P[P["sample"] == "discovery"]), ("confirmation", P[P["sample"] == "confirmation"]), ("west", P[P.west == 1]), ("east", P[P.west == 0])]})
    S.round(3).to_csv(f"{DESC}/usw_summary_statistics.csv"); S.round(3).to_latex(f"{TAB}/usw_summary_statistics.tex")
    ex = P.groupby(["sample", "west"]).agg(hospitals=("hospital_pk", "nunique"), hospital_weeks=("w1", "size"), mean_w1=("w1", "mean"), severe_weeks=("w4", "sum"), share_severe=("w4", "mean"), weeks_w1_ge10=("w1", lambda s: int((s >= 10).sum())), weeks_w1_ge25=("w1", lambda s: int((s >= 25).sum())))
    ex.round(4).to_csv(f"{DESC}/usw_exposure_by_sample_region.csv")
    H = P.drop_duplicates("hospital_pk"); H.groupby("west").sev_days_0619.describe().round(2).to_csv(f"{DESC}/usw_experience_by_region.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    w = P.groupby(["week", "west"]).w1.mean().unstack(); ax[0].plot(w.index, w[0], color=C1, label="East"); ax[0].plot(w.index, w[1], color=C2, label="West"); ax[0].axvline(pd.Timestamp(DISC_END), color=C3, ls="--", lw=.8)
    ax[0].set_title("Weekly mean smoke PM2.5 at hospitals"); ax[0].set_ylabel("µg/m³"); ax[0].legend(frameon=False); ax[0].tick_params(axis="x", rotation=30)
    x = P.w1.dropna(); ax[1].hist(x[x > 0].clip(upper=60), bins=60, color=C1); ax[1].set_yscale("log"); ax[1].set_title(f"Hospital-weeks with smoke PM > 0 ({(x > 0).mean():.0%} of all)"); ax[1].set_xlabel("weekly mean smoke PM2.5 (capped at 60)")
    ax[2].hist([H[H.west == 0].sev_days_0619.clip(upper=80), H[H.west == 1].sev_days_0619.clip(upper=80)], bins=30, color=[C1, C2], label=["East", "West"], stacked=True); ax[2].set_title("Prior experience: days ≥ 20 µg/m³, 2006–2019"); ax[2].set_xlabel("days (capped at 80)"); ax[2].legend(frameon=False)
    save(fig, "fig01_exposure_distributions")

def coefplot():
    R = pd.read_csv(f"{ECON}/usw_all_results_adjusted.csv"); m = R[(R.spec == "main") & (R.exposure == "w1") & R.outcome.isin(["occ", "icu_occ", "l_ed"])]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2)); lab = {"occ": "Inpatient occupancy (pp)", "icu_occ": "ICU occupancy (pp)", "l_ed": "log ED visits ×100"}
    for a, term, ttl in ((ax[0], "beta", "Smoke effect at mean experience (per 10 µg/m³)"), (ax[1], "delta", "Smoke × prior experience (per SD)")):
        for j, y in enumerate(lab):
            for k, (s, c) in enumerate((("discovery", C1), ("confirmation", C2), ("pooled", C3))):
                r = m[(m.outcome == y) & (m.term == term) & (m["sample"] == s)].iloc[0]; sc = 100 if y == "l_ed" else 1
                a.errorbar(r.coef * sc, j + (k - 1) * .22, xerr=1.96 * r.se * sc, fmt="o", color=c, capsize=2, label=s if j == 0 else None)
        a.axvline(0, color="k", lw=.6); a.set_yticks(range(3)); a.set_yticklabels(lab.values()); a.set_title(ttl)
    ax[0].legend(frameon=False, fontsize=8); save(fig, "fig02_main_effects_discovery_vs_confirmation")

def dlag():
    D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ECON}/usw_*_distributed_lag.csv")]); D = D[(D.exposure == "w1") & (D.interacted == False)]
    order = ["w1_lead2", "w1_lead1", "w1", "w1_lag1", "w1_lag2", "w1_lag3", "w1_lag4"]; fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a, (y, t) in zip(ax, (("occ", "Inpatient occupancy (pp)"), ("icu_occ", "ICU occupancy (pp)"), ("l_ed", "log ED visits"))):
        for s, c, off in (("discovery", C1, -.12), ("confirmation", C2, .12)):
            d = D[(D.outcome == y) & (D["sample"] == s)].set_index("term").reindex(order); a.errorbar(np.arange(-2, 5) + off, d.coef, yerr=1.96 * d.se, fmt="o-", color=c, capsize=2, lw=.8, label=s)
        a.axhline(0, color="k", lw=.6); a.axvline(-.5, color=C3, ls=":"); a.set_title(t); a.set_xlabel("weeks relative to exposure (negative = future smoke)")
    ax[0].legend(frameon=False); save(fig, "fig03_distributed_lag_leads_and_lags")

def evstudy():
    E = pd.read_csv(f"{ECON}/usw_stacked_event_study.csv"); fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a, (y, t) in zip(ax, (("occ", "Inpatient occupancy (pp)"), ("icu_occ", "ICU occupancy (pp)"), ("l_ed", "log ED visits"))):
        for g, c, off in (("all", "k", 0), ("treated_low_experience", C1, -.15), ("treated_high_experience", C2, .15)):
            d = E[(E.outcome == y) & (E.group == g)].sort_values("k"); a.errorbar(d.k + off, d.coef, yerr=1.96 * d.se, fmt="o-", color=c, capsize=2, lw=.8, ms=3, label=g.replace("treated_", "").replace("_", " "))
        a.axhline(0, color="k", lw=.6); a.axvline(-.5, color=C3, ls=":"); a.set_title(t); a.set_xlabel("weeks from first severe smoke week (ref. = −1)")
    ax[0].legend(frameon=False, fontsize=7); save(fig, "fig04_stacked_event_study")

def robust():
    R = pd.read_csv(f"{ECON}/usw_all_results_adjusted.csv"); r = R[(R.outcome == "occ") & (R.exposure == "w1") & (R.family.isin(["robust", "primary", "alt_experience"])) & (R.spec != "E1_terciles")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    specs = sorted(r.spec.unique())
    for a, term, ttl in ((ax[0], "beta", "β: smoke effect on occupancy (pp per 10 µg/m³)"), (ax[1], "delta", "δ: smoke × experience")):
        for s, c, off in (("discovery", C1, -.15), ("confirmation", C2, .15)):
            d = r[(r.term == term) & (r["sample"] == s)].drop_duplicates("spec").set_index("spec").reindex(specs); a.errorbar(d.coef, np.arange(len(specs)) + off, xerr=1.96 * d.se, fmt="o", color=c, capsize=1.5, ms=3, label=s)
        a.axvline(0, color="k", lw=.6); a.set_title(ttl); a.set_yticks(range(len(specs))); a.set_yticklabels(specs, fontsize=7)
    ax[0].legend(frameon=False); save(fig, "fig05_robustness_summary_occupancy")

def quebec():
    Q = pd.read_csv(f"{ECON}/qcd_results.csv"); d = Q[(Q.spec == "dist_lag") & Q.term.str.startswith("spm10")]
    order = ["spm10_lead3", "spm10_lead2", "spm10_lead1", "spm10"] + [f"spm10_lag{k}" for k in range(1, 8)]; fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a, (y, t) in zip(ax, (("occ", "Stretcher occupancy (pp)"), ("l_waiting", "log patients waiting"), ("l_p24", "log patients > 24 h on stretcher"))):
        x = d[d.outcome == y].set_index("term").reindex(order); a.errorbar(np.arange(-3, 8), x.coef, yerr=1.96 * x.se, fmt="o-", color=C1, capsize=2, lw=.8); a.axhline(0, color="k", lw=.6); a.axvline(-.5, color=C3, ls=":"); a.set_title("Quebec EDs: " + t); a.set_xlabel("days relative to exposure")
    save(fig, "fig06_quebec_distributed_lag")
    P = pd.read_parquet(f"{PROC}/qc_panel.parquet", columns=["date", "occ", "spm"]); g = P[(P.date >= "2023-05-01") & (P.date <= "2023-08-15")].groupby("date").agg(occ=("occ", "mean"), spm=("spm", "mean"))
    fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(g.index, g.occ, color=C1); ax.set_ylabel("mean stretcher occupancy (%)", color=C1); a2 = ax.twinx(); a2.spines["right"].set_visible(True); a2.fill_between(g.index, g.spm, color=C2, alpha=.35); a2.set_ylabel("mean smoke PM2.5 (µg/m³)", color=C2); ax.set_title("Quebec EDs, summer 2023")
    save(fig, "fig07_quebec_2023_occupancy_and_smoke")

def ml():
    M = pd.read_csv(f"{ML}/metrics_occ_h1.csv"); t = M[(M.period == "test")]
    keep = [m for m in t.model.unique() if m.split("_")[-1] in ("C", "AR") or m in ("persistence", "hist_mean", "seasonal_naive", "rolling_mean4")]
    a_ = t[(t.regime == "all") & t.model.isin(keep)].sort_values("r2_vs_persistence"); fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a_ = a_[a_.r2_vs_persistence > -0.5]; ax[0].barh(a_.model, a_.r2_vs_persistence, color=C1); ax[0].axvline(0, color="k", lw=.6); ax[0].set_title("Out-of-sample R² vs persistence, 2023–24 test period\n(1-week-ahead inpatient occupancy)")
    p = t[t.regime.isin(["normal", "smoke", "post_smoke"]) & t.model.isin(["ar_AR", "ridge_C", "lgbm_A", "lgbm_B", "lgbm_C", "xgb_C", "lgbm_Bretro"])].pivot(index="model", columns="regime", values="rmse")[["normal", "smoke", "post_smoke"]]
    p.plot.bar(ax=ax[1], color=[C3, C2, C1], width=.75); ax[1].set_title("RMSE by regime (pp of occupancy)"); ax[1].set_ylim(p.values.min() * .9, p.values.max() * 1.03); ax[1].tick_params(axis="x", rotation=30); ax[1].legend(frameon=False)
    save(fig, "fig08_forecast_model_comparison")
    I = pd.read_csv(f"{ML}/incremental_occ_h1.csv"); i = I[(I.period == "test") & I.regime.isin(["all", "smoke"]) & I.learner.isin(["ols", "ridge", "lgbm", "xgb"])]
    i["step"] = i.small + "→" + i.big; pv = i.pivot_table(index=["learner", "step"], columns="regime", values="r2_big_vs_small"); fig, ax = plt.subplots(figsize=(8, 4.2)); (100 * pv).plot.barh(ax=ax, color=[C1, C2]); ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("change in out-of-sample MSE explained (%), larger information set vs smaller"); ax.set_title("Incremental value: smoke (A→B), experience (B→C), capacity (C→D), realised smoke (retro)", fontsize=9); ax.legend(frameon=False)
    save(fig, "fig09_incremental_value_of_information_sets")
    F = pd.read_parquet(f"{ML}/eval_frame_occ_h1.parquet"); T = F[F.period == "test"]; fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    s = T.sample(15000, random_state=SEED); ax[0].scatter(s.lgbm_C, s.target, s=1, alpha=.25, color=C1); ax[0].plot([0, 120], [0, 120], color="k", lw=.6); ax[0].set_xlabel("predicted (LightGBM, set C)"); ax[0].set_ylabel("actual occupancy (%)"); ax[0].set_title("Actual vs predicted, test period")
    g = T.groupby("target_week").agg(a=("target", "mean"), p=("lgbm_C", "mean"), b=("persistence", "mean")); ax[1].plot(g.index, g.a, color="k", label="actual"); ax[1].plot(g.index, g.p, color=C1, ls="--", label="LightGBM C"); ax[1].plot(g.index, g.b, color=C3, ls=":", label="persistence"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("Mean occupancy across hospitals"); ax[1].tick_params(axis="x", rotation=30)
    e = pd.DataFrame({"regime": T.regime, "LightGBM C": (T.target - T.lgbm_C).abs(), "persistence": (T.target - T.persistence).abs(), "AR": (T.target - T.ar_AR).abs()}).groupby("regime").mean().reindex(["normal", "smoke", "post_smoke"]); e.plot.bar(ax=ax[2], color=[C1, C3, C2], width=.75); ax[2].set_title("Mean absolute error by regime"); ax[2].tick_params(axis="x", rotation=0); ax[2].legend(frameon=False, fontsize=7)
    save(fig, "fig10_actual_vs_predicted_and_errors_by_regime")
    t2 = t[t.model.isin(["ar_AR", "lgbm_A", "lgbm_C"]) & t.regime.isin(["smoke_E_low", "smoke_E_high", "E_low", "E_high"])].pivot(index="regime", columns="model", values="r2_vs_persistence").reindex(["E_low", "E_high", "smoke_E_low", "smoke_E_high"])
    fig, ax = plt.subplots(figsize=(6, 3.2)); t2.plot.bar(ax=ax, color=[C3, C1, C2], width=.75); ax.set_title("Out-of-sample R² vs persistence by prior-experience tercile"); ax.tick_params(axis="x", rotation=0); ax.legend(frameon=False, fontsize=7); save(fig, "fig11_forecast_skill_by_experience")

for f in (desc, coefplot, dlag, evstudy, robust, quebec, ml):
    attempt(f)
