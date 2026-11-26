"""Recompute the headline forecast metrics from the tracked test-period predictions. Runs in a few seconds.

Input : results/forecasting/sample_predictions_occ_h1.parquet
        (one-week-ahead inpatient occupancy forecasts, 1,000 hospitals, target weeks January 2023 to April 2024)
Usage : python examples/reproduce_headline_metrics.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = pd.read_parquet(os.path.join(ROOT, "results", "forecasting", "sample_predictions_occ_h1.parquet"))

MODELS = {
    "persistence": "No-change forecast (occupancy next week = occupancy this week)",
    "ar_AR": "Pooled autoregression (4 lags + hospital mean)",
    "ridge_C": "Ridge, lags + calendar + weather + smoke + smoke history",
    "lgbm_A": "LightGBM, lags + calendar only",
    "lgbm_B": "LightGBM, + weather and smoke known at forecast time",
    "lgbm_C": "LightGBM, + smoke history of the county",
}


def table(d):
    mse = {m: float(((d.target - d[m]) ** 2).mean()) for m in MODELS}
    rows = [dict(model=m, description=MODELS[m], rmse=np.sqrt(mse[m]), mae=float((d.target - d[m]).abs().mean()),
                 mse_reduction_vs_no_change=1 - mse[m] / mse["persistence"], mse_reduction_vs_autoregression=1 - mse[m] / mse["ar_AR"]) for m in MODELS]
    return pd.DataFrame(rows).set_index("model")


pd.set_option("display.width", 220)
print(f"Hospital-weeks: {len(P):,}   hospitals: {P.hospital_pk.nunique():,}   target weeks: {P.target_week.min():%Y-%m-%d} to {P.target_week.max():%Y-%m-%d}")
print("\nAll test weeks (occupancy in percentage points)")
print(table(P).drop(columns="description").round(4).to_string())
smoke = P[P.regime == "smoke"]
print(f"\nSevere-smoke target weeks only (n = {len(smoke):,})")
print(table(smoke).drop(columns="description").round(4).to_string())
a, b = table(P), table(smoke)
print("\nChange in mean squared error from adding weather and smoke features to LightGBM (negative = better):")
print(f"  all weeks   {100 * (a.rmse['lgbm_B'] ** 2 / a.rmse['lgbm_A'] ** 2 - 1):+.2f}%")
print(f"  smoke weeks {100 * (b.rmse['lgbm_B'] ** 2 / b.rmse['lgbm_A'] ** 2 - 1):+.2f}%")
