"""Summary figure: squared-error reduction against the no-change forecast, by model (test period, occupancy, one week ahead).
Reads results/forecasting/metrics_occ_h1.csv. Output: reports/figures/fig_summary_forecast_gain.png
"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
M = pd.read_csv(f"{ML}/metrics_occ_h1.csv"); M = M[(M.period == "test") & (M.regime == "all")].set_index("model")
ROWS = [("xgb_A", "XGBoost", 1), ("lgbm_A", "LightGBM", 1), ("gboost_A", "Gradient boosting", 1), ("catboost_A", "CatBoost", 1), ("rf_A", "Random forest", 1),
        ("mlp_A", "Neural network", 1), ("ridge_A", "Ridge regression", 1), ("ar_AR", "Autoregression", 0), ("rolling_mean4", "4-week average", 0)]
BLUE, GREY, INK, MUTED = "#2f6aa6", "#9aa0a8", "#22252a", "#5f6670"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 15})
fig, ax = plt.subplots(figsize=(6.4, 5.6))
vals = [100 * M.r2_vs_persistence[k] for k, _, _ in ROWS]; y = list(range(len(ROWS)))[::-1]
ax.barh(y, vals, height=0.66, color=[BLUE if m else GREY for _, _, m in ROWS])
for yi, v in zip(y, vals):
    ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", ha="left", color=INK, fontsize=14)
ax.set_yticks(y); ax.set_yticklabels([n for _, n, _ in ROWS], color=INK)
ax.axhline(1.5, color="#d9d5cc", lw=1, ls=(0, (4, 3)))
ax.set_xlim(0, 24.5); ax.set_xticks([0, 5, 10, 15, 20]); ax.set_xticklabels(["0", "5", "10", "15", "20%"], color=MUTED)
ax.set_xlabel("Reduction in squared forecast error\nvs a no-change forecast", color=MUTED, fontsize=13.5)
ax.grid(axis="x", color="#ebe8e1", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#cfcac0"); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", color="#cfcac0")
ax.set_title("Next-week hospital bed occupancy:\nforecast gain by model", loc="left", color=INK, fontsize=16.5, fontweight="bold", pad=12)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_summary_forecast_gain.png", dpi=200, facecolor="white"); print([round(v, 2) for v in vals])
