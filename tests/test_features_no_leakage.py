"""The forecast feature builder must only use information available at the forecast origin.
These tests run on a small synthetic hospital-week panel; no project data are needed."""
import numpy as np
import pandas as pd
import pytest

import features

H = 1
N_HOSP, N_WEEKS = 12, 80


def synthetic_panel(seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2020-08-02", periods=N_WEEKS, freq="7D")
    rows = []
    for i in range(N_HOSP):
        base = rng.uniform(40, 85)
        occ = base + np.cumsum(rng.normal(0, 2, N_WEEKS))
        for t, w in enumerate(weeks):
            rows.append(dict(hospital_pk=f"H{i:03d}", week=w, state="AA" if i % 2 else "BB", division="9_Pacific", west=1, hospital_subtype="Short Term",
                             is_metro_micro=True, lat=40.0 + i, lon=-120.0 + i, occ=occ[t], icu_occ=occ[t] - 5, ed=500 + 10 * t, beds=200.0, icu_beds=20.0,
                             covid_share=0.05, occ_cov=1.0, w1=rng.exponential(2), w2=0.0, w3=0.0, w4=float(rng.random() < 0.05), w5=0.0, w1_lag1=0.0, w1_lag2=0.0,
                             w2_lag1=0.0, w1_cum4=0.0, tmax=20.0, prcp=1.0, E1=float(i) / N_HOSP, E1_terc=1 + i % 3, yrs_since_last_sev=3.0, E2=0.1, E6=0.2,
                             med_income=60000.0, poverty=12.0, share65=0.16, places_copd=6.0, pop=100000.0, fema_fire_pm4=0.0))
    return pd.DataFrame(rows)


def build(tmp_path, monkeypatch, panel, name):
    d = tmp_path / name
    d.mkdir()
    panel.to_parquet(d / "analysis_panel.parquet", index=False)
    monkeypatch.setattr(features, "PROC", str(d))
    return features.build("occ", H, "occ_cov")


def test_target_is_the_outcome_h_weeks_after_the_origin(tmp_path, monkeypatch):
    panel = synthetic_panel()
    F = build(tmp_path, monkeypatch, panel, "a")
    truth = panel.set_index(["hospital_pk", "week"]).occ
    assert ((F.target_week - F.week).dt.days == 7 * H).all()
    expected = truth.reindex(pd.MultiIndex.from_frame(F[["hospital_pk", "target_week"]])).values
    assert np.allclose(F.target.values, expected, atol=1e-3)
    assert np.allclose(F.y_t.values, truth.reindex(pd.MultiIndex.from_frame(F[["hospital_pk", "week"]])).values, atol=1e-3)


def test_real_time_feature_sets_contain_no_target_week_columns():
    for name in ("A", "B", "C", "D"):
        assert not [c for c in features.SETS[name] if c.endswith("_retro") or c.endswith("r")], name
    assert "target" not in features.SETS["D"] and "target_week" not in features.SETS["D"]


def test_features_at_an_origin_do_not_change_when_the_future_changes(tmp_path, monkeypatch):
    panel = synthetic_panel()
    cut = sorted(panel.week.unique())[50]
    future = panel.copy()
    later = future.week > cut
    for c in ("occ", "icu_occ", "ed", "w1", "w4", "tmax", "prcp", "covid_share", "beds"):
        future.loc[later, c] = future.loc[later, c] * 3 + 7
    a = build(tmp_path, monkeypatch, panel, "base")
    b = build(tmp_path, monkeypatch, future, "changed")
    cols = features.SETS["D"]
    a = a[a.week <= cut].set_index(["hospital_pk", "week"])[cols].sort_index()
    b = b[b.week <= cut].set_index(["hospital_pk", "week"])[cols].sort_index()
    assert len(a) > 0 and a.index.equals(b.index)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-4)


def test_subsample_is_reproducible(tmp_path, monkeypatch):
    F = build(tmp_path, monkeypatch, synthetic_panel(), "s")
    one = features.subsample(F, n=5).hospital_pk.unique()
    two = features.subsample(F, n=5).hospital_pk.unique()
    assert len(one) == 5 and list(one) == list(two)
