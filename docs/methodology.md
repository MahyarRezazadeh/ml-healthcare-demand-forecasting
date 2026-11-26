# Methodology

## 1. Unit and targets

One observation is a hospital-week. The forecast is made at the end of week *t* for week *t + h*.

| Target | Definition | Horizons |
|---|---|---|
| Inpatient occupancy | occupied inpatient beds / staffed inpatient beds × 100 (7-day averages) | 1 and 4 weeks |
| ICU occupancy | occupied ICU beds / staffed ICU beds × 100 | 1 week |
| ED visits | log(1 + total ED visits in the week) | 1 week |

Models are fitted to the change `y(t+h) − y(t)`. The level forecast is `y(t)` plus the predicted change, so the no-change forecast is the special case of a zero predicted change.

## 2. Feature sets

Feature sets are nested so that the value of each block of information can be measured (`src/features.py`).

| Set | Adds | Examples |
|---|---|---|
| A | history and calendar | current value, lags 1, 2, 3, 7; 4- and 8-week rolling mean; 8-week rolling SD; same week last year; hospital expanding mean; week-of-year harmonics; location; COVID share; log beds |
| B | weather and smoke known at *t* | weekly mean smoke PM2.5, heavy-plume days, plume days, severe-week flag, weekly maximum, lags, 4-week cumulative smoke, maximum temperature, precipitation |
| C | county smoke history 2006–2019 | standardised count of severe smoke days, tercile, years since last severe year, interactions with current smoke |
| D | baseline capacity and county characteristics | baseline beds, ICU beds and occupancy; income, poverty, share aged 65+, COPD prevalence, population |
| B-retro, C-retro | smoke and weather **realised in the target week** | used only as an upper bound on what a perfect smoke forecast could add; never reported as a forecast |

## 3. Leakage controls

- Every feature in sets A–D is computed from data dated week *t* or earlier. `tests/test_features_no_leakage.py` changes all values after a cut-off week and checks that features at earlier origins are unchanged.
- A training row is used at origin *t* only if its target week is already observed (`t_idx + h <= t`).
- Baseline capacity variables use the first 12 weeks of the window, before the first forecast origin.
- Hyperparameters are tuned on data before 2023 only and then frozen for every target, horizon and feature set.

## 4. Models

Benchmarks: no-change (persistence), historical mean, seasonal naive, 4-week rolling mean, pooled autoregression with and without seasonal terms.

Learners: OLS, ridge, lasso, elastic net, decision tree, random forest, extra trees, histogram gradient boosting, LightGBM, XGBoost, CatBoost, multilayer perceptron, support vector regression, and a non-negative stacking ensemble. Additional single-split comparisons: ARIMA, SARIMAX with smoke, Prophet, Prophet + XGBoost (150 hospitals), LSTM, GRU, TCN, and Poisson regression for ED counts.

Tuning (`src/10_tune_hyperparameters.py`): random search, 8 configurations per learner, trained on targets to December 2021 and validated on 2022. Chosen values are in `results/forecasting/tuned_hyperparameters.json`.

## 5. Walk-forward design

`src/11_forecast_walkforward.py` uses an expanding window. The first forecast origin is week 52. At each refit the model is trained on all rows whose targets are observed, then predicts every later origin until the next refit.

| Learners | Refit every |
|---|---|
| OLS, ridge, LightGBM, XGBoost, autoregression | 4 weeks |
| lasso, elastic net | 13 weeks |
| tree, forests, CatBoost, MLP, SVR | 26 weeks, training sample capped at 100,000 rows (SVR 12,000) |

Target weeks from January 2023 are the **test period**. Earlier forecast weeks are the tuning period.

## 6. Evaluation (`src/12_evaluate_forecasts.py`)

- **Accuracy:** RMSE, MAE, bias, MASE, and out-of-sample R² against the no-change forecast and against the autoregression (1 − MSE ratio).
- **Regimes:** all weeks, severe-smoke target weeks, weeks after smoke, by tercile of county smoke history, by capacity, East and West.
- **Tests:** Diebold–Mariano tests with standard errors clustered two-way by state and week; Clark–West test for nested OLS models; week-block bootstrap intervals for R².
- **Incremental value:** same learner and same observations, feature set B against A, C against B, D against C, and the retrospective sets against the real-time sets.
- **Intervals:** split-conformal 90% intervals from tuning-period absolute errors, checked for coverage in the test period.
- **Alerts:** detection of weeks above the hospital's own 90th percentile of pre-2023 occupancy (AUROC, PR-AUC, precision, recall, F1).

Further validation designs (`src/13_forecast_validation_designs.py`): 104-week rolling window, leave-one-census-division-out, 5-fold leave-hospital-out, equal-sized training samples by smoke-history group, leave-one-smoke-quarter-out.

## 7. Does smoke change occupancy? (causal design)

The forecasting test asks whether smoke *predicts* occupancy. A separate set of regressions asks whether smoke *changes* it.

- **Specification** (`src/20_panel_regressions.py`): outcome on weekly mean smoke PM2.5 and its interaction with county smoke history, with hospital fixed effects, census-division-by-week fixed effects, state-by-calendar-month fixed effects, and weather controls (temperature quintiles and precipitation). Hospital-weeks near a fire declaration in the county are excluded so that the variation comes from transported smoke. Standard errors are clustered by state and week.
- **Discovery and held-out samples:** the specification was written down before outcome data were opened ([protocol](protocol/research_protocol_v1.md)). It was estimated on data to December 2022 and then run unchanged on January 2023 to April 2024.
- **Checks:** leads and lags, a stacked event study around the first severe smoke week, a permutation placebo, alternative exposure and history measures, Holm and Benjamini–Hochberg corrections (`src/22_event_study_placebo.py`, `src/24_results_tables.py`).
- **Double machine learning** (`src/23_double_ml_causal_forest.py`): partially linear model with LightGBM nuisance functions and state-grouped cross-fitting, and a causal forest for heterogeneity on a 500-hospital sample.

## 8. Reproducibility

- Seed `20261125` in `src/config.py` fixes the hospital sample, the tuning draws and every model.
- Package versions are pinned in `requirements.txt`.
- The forecasting scripts write one file per model and feature set and skip finished files, so interrupted runs resume.
- Deviations from the written protocol are listed in [protocol/protocol_amendments.md](protocol/protocol_amendments.md) and [limitations_and_next_steps.md](limitations_and_next_steps.md). The protocol files use the original script names; the mapping is in the table below.

| Name in the protocol | Name in `src/` |
|---|---|
| `05_hms_points.py` | `06_overlay_smoke_plumes.py` |
| `06_build_covariates.py` | `07_build_covariates.py` |
| `07_build_exposure.py` | `08_build_analysis_panel.py` |
| `09_build_quebec.py` | `09_build_quebec_panel.py` |
| `12_ml_tune.py` | `10_tune_hyperparameters.py` |
| `11_ml_core.py` | `11_forecast_walkforward.py` |
| `14_ml_eval.py` | `12_evaluate_forecasts.py` |
| `17_ml_extra.py` | `13_forecast_validation_designs.py` |
| `18_ml_quebec.py` | `14_forecast_quebec_daily.py` |
| `08_econometrics.py` | `20_panel_regressions.py` |
| `10_econ_quebec.py` | `21_panel_regressions_quebec.py` |
| `16_event_study.py` | `22_event_study_placebo.py` |
| `15_causal_ml.py` | `23_double_ml_causal_forest.py` |
| `19_tables_econ.py` | `24_results_tables.py` |
| `20_figures.py` | `30_figures.py` |
| `ml_data.py` | `features.py` |

Scripts 13 and 21–23 of the original numbering produced a literature review and journal assessment and are not part of this repository.
