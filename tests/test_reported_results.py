"""The tracked sample of predictions must reproduce the tracked metrics table."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "forecasting")


def test_sample_predictions_reproduce_metrics_table():
    P = pd.read_parquet(os.path.join(RES, "sample_predictions_occ_h1.parquet"))
    M = pd.read_csv(os.path.join(RES, "metrics_occ_h1.csv"))
    M = M[(M.period == "test") & (M.regime == "all")].set_index("model")
    for m in ("persistence", "ar_AR", "ridge_C", "lgbm_A", "lgbm_B", "lgbm_C"):
        rmse = float(np.sqrt(((P.target - P[m]) ** 2).mean()))
        assert abs(rmse - M.rmse[m]) < 1e-4, (m, rmse, M.rmse[m])
        assert int(M.n[m]) == len(P)


def test_boosted_trees_beat_both_benchmarks_and_smoke_adds_nothing():
    M = pd.read_csv(os.path.join(RES, "metrics_occ_h1.csv"))
    M = M[(M.period == "test") & (M.regime == "all")].set_index("model")
    assert M.rmse["lgbm_A"] < M.rmse["ar_AR"] < M.rmse["persistence"]
    assert M.rmse["lgbm_B"] >= M.rmse["lgbm_A"]          # weather + smoke features do not lower the error
