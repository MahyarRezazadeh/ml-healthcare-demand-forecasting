# Research protocol v1 — Wildfire smoke, hospital congestion, and prior exposure

**Written:** 2026-11-25, before any outcome data were opened or any outcome–exposure relationship was estimated.
At the time of writing only dataset metadata (column names, row counts, date ranges) had been inspected.
**Rule:** this file is frozen. Any change is recorded as a dated entry in `protocol_amendments.md` with the reason and whether results had been seen.

## 1. Questions and hypotheses

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Smoke raises respiratory / total acute demand (ED visits, admissions) | secondary where data allow |
| H2 | Smoke raises hospital congestion (inpatient occupancy, ICU occupancy, ED volume, stretcher occupancy) | **primary** |
| H3 | The congestion response to a given smoke shock is smaller where prior smoke exposure is greater (δ < 0 when β > 0) | **primary** |
| H4 | Faster recovery (shorter lag profile) where prior exposure is greater | secondary |
| H5 | Smoke displaces non-respiratory / non-COVID care or shifts admission–discharge patterns | secondary |
| H6 | Forecast accuracy differs between normal and smoke periods | **primary (ML)** |
| H7 | During smoke periods outcomes are more predictable in high-experience areas | **primary (ML)** |
| H8 | Smoke information and prior-experience features add out-of-sample predictive value beyond history, calendar and hospital characteristics | **primary (ML)** |
| H9 | Hospitals near capacity adjust beds / admissions more | secondary |
| H10 | Differences by experience reflect patient behaviour, demographics, baseline capacity or training-sample size rather than hospital learning | tested as competing explanations |

No direction is assumed. A null or opposite-signed result is reported as such.

## 2. Contexts

**Context US-W (primary): US hospital-week panel.** HHS "COVID-19 Reported Patient Impact and Hospital Capacity by Facility" (healthdata.gov `anag-cw7u`), weekly, facility level. Analysis window: collection weeks 2020-08-01 to 2024-04-27 (the mandatory federal reporting period). Linked to hospital coordinates.

**Context US-D (secondary): US state-day panel.** HHS state time series (daily) — used for the daily forecasting horizons (1, 3, 7 days).

**Further contexts** (Canada, Australia, UK, others) are added only after the data audit shows a hospital outcome series that was actually downloaded. Each addition is logged as an amendment *before* its outcomes are analysed, using the same specification template below.

## 3. Sample rules (US-W)

- Suppressed cells (coded −999999, true value 1–3) are set to missing; no imputation in the main analysis.
- Hospital subtypes: short-term acute care and critical access hospitals. Primary sample = short-term acute care.
- A hospital enters if it has non-missing outcome in ≥ 80 % of weeks in the window and a valid location.
- Occupancy ratios outside [0, 1.5] are set to missing (reporting errors).
- CONUS only (the smoke products cover CONUS reliably).

## 4. Outcomes

Primary family (confirmatory):
- **P1** Inpatient bed occupancy rate = `inpatient_beds_used_7_day_avg / inpatient_beds_7_day_avg` (percentage points).
- **P2** Total ED visits = `previous_day_total_ed_visits_7_day_sum` (log(1+y) in linear models; PPML on the count as the alternative).
- **P3** ICU occupancy rate = `icu_beds_used_7_day_avg / total_icu_beds_7_day_avg` (percentage points).

Secondary: occupied inpatient beds (level/log); non-COVID occupied inpatient beds (`used − used_covid`); non-COVID ED visits (`total − covid ED`); adult inpatient occupancy; paediatric occupied beds; staffed inpatient beds and ICU beds (supply response, H9); cumulative excess occupancy over lags 0–4 (H4/H5).

Outcomes are named for what the source measures. Occupancy is not called waiting time; changes in occupied beds are not called discharges; post-smoke rebounds are not called postponed treatment.

## 5. Exposure W

- **W1 (primary):** weekly mean wildfire-smoke PM2.5 (µg/m³) at the hospital, built from EPA AQS daily PM2.5 monitors and NOAA HMS plumes: smoke PM on day d at monitor m = max(0, PM − median PM of that monitor in that calendar month on non-plume days in the preceding three calendar years) if an HMS plume is overhead, else 0; hospital value = inverse-distance-weighted mean of monitors within 100 km. Hospitals with no monitor within 100 km are excluded from W1 models.
- **W2:** days in the week with an HMS *heavy* plume over the hospital (0–7). All hospitals.
- **W3:** days with any HMS plume. **W4:** severe-week indicator (weekly mean smoke PM ≥ 15 µg/m³ or ≥ 2 heavy-plume days). **W5:** weekly maximum daily smoke PM. **W6:** consecutive smoke days / cumulative exposure over the previous 4 weeks.
- The W1 construction is validated against the Stanford Echo Lab county smoke-PM product for 2018–2020 before use; the correlation is reported whatever it is.
- Total ambient PM2.5 is kept separate from smoke PM2.5.
- Transported vs local: main specification drops county-weeks with a FEMA fire declaration (FM, or DR/EM with incident type Fire) within ±4 weeks. A separate eastern-US 2023 sample isolates long-range Canadian smoke.

## 6. Prior experience E (predetermined)

Built from the Stanford county daily smoke-PM product, **2006-01-01 to 2019-12-31 only** (ends before the outcome window opens).
- **E1 (primary, continuous):** asinh(number of days with county smoke PM ≥ 20 µg/m³, 2006–2019), standardised.
- E2: mean annual smoke PM 2006–2019. E3: number of distinct episodes (runs of days ≥ 20 µg/m³ separated by ≥ 7 clean days). E4: the same counts over the last 1, 3, 5, 10 pre-sample years. E5: years since the last day ≥ 20 µg/m³ as of 2019-12-31. E6: maximum daily smoke PM 2006–2019. E7 (time-varying): E1 updated with in-sample exposure up to week t−8 (W2-based), never including the current episode.
- Categorical version (secondary): terciles of E1; "limited historical exposure within the observed period" = bottom tercile. No hospital is called first-time exposed.

## 7. Econometric specifications

Baseline: `Y_it = α_i + λ_{d(i),t} + μ_{s(i),m(t)} + β W_it + δ (W_it × E_i) + X_it'γ + ε_it`
- α_i hospital FE; λ census-division × week FE; μ state × calendar-month FE. E_i is absorbed by α_i; E7 enters with its main effect.
- X: county weekly mean temperature (5 bins of weekly mean tmax), precipitation, federal-holiday week. No COVID control in the main model (smoke may affect COVID outcomes); lagged COVID inpatient share is an alternative specification.
- Standard errors: two-way clustered by state and week (primary); alternatives: state only; HHS region × week; county.
- Count outcomes: PPML with the same FE.
- Alternative FE: hospital + week; hospital + state × week.

Dynamics: distributed-lag model with two leads and four lags of W (and of W × E). Leads are the pre-trend/anticipation test; the sum of lags 0–4 is the cumulative effect; lags 1–4 test persistence, recovery and displacement. A stacked episode design (episode = first week with W4 = 1 after ≥ 4 clean weeks; controls = same-division hospitals with no W4 week within ±6 weeks) is the check for overlapping events.

Identification statement fixed in advance: β is identified from within-hospital, within-division-week variation in plume-driven smoke. δ is a **heterogeneous association** unless it survives (i) W × baseline-characteristic interactions (beds, baseline occupancy, metro status, West indicator, county income, share 65+, baseline smoke-season PM), (ii) estimation within the West and within the East separately, and (iii) the confirmation sample. Even then δ is evidence on experience-dependent congestion, not proof of institutional learning.

Placebos: leads of W; smoke series shifted by 52 weeks; within-division permutation of smoke histories across hospitals (500 draws).

## 8. Discovery / confirmation split

- **Discovery:** weeks 2020-08-01 to 2022-12-31.
- **Confirmation (held out):** weeks 2023-01-01 to 2024-04-27. It contains the 2023 Canadian smoke episodes that reached eastern states. Confirmation estimates are run once, after the specification list in this file is executed on the discovery sample, with no specification changes.
- Pooled full-window estimates are reported and labelled as pooled.
- If the two samples disagree, the result is classified as not confirmed.

## 9. Multiple testing

- Confirmatory set: β and δ for P1, P2, P3 under W1 (six tests) — Holm family-wise correction.
- Benjamini–Hochberg FDR within each family: (a) main effects across all outcomes × W1–W2; (b) interactions across all outcomes × E1; (c) forecast-comparison tests.
- Unadjusted and adjusted p-values, confidence intervals, N, and the number of independent smoke episodes are always reported.

## 10. Machine learning

Target: outcome at t+h, h = 1 and 4 weeks (US-W); 1, 3, 7 days (US-D).
Feature sets: **A** lagged outcomes (1–4, 8, 52 where available), rolling means, calendar, hospital characteristics; **B** = A + weather + smoke (lags of W1–W3 known at the forecast origin); **C** = B + E and E × smoke; **D** = C + capacity (beds, ICU beds, baseline occupancy), county vulnerability (ACS).
Two variants of B–D: *real-time* (smoke up to the forecast origin only) and *retrospective* (adds realised smoke in the target week; an upper bound for a perfect smoke forecast, never called a forecast).

Models: historical mean, seasonal mean, seasonal naive, persistence, pooled AR, OLS, ARIMA/SARIMA(X) and Prophet (on a random hospital subsample or state aggregates if per-hospital fitting is infeasible), Poisson; ridge, lasso, elastic net; decision tree, random forest, extra trees, gradient boosting, XGBoost, LightGBM, CatBoost, SVR (subsample if infeasible at full N); MLP, LSTM, GRU, TCN, Prophet-XGBoost hybrid, stacking — only where the software installs and the sample supports them. Every skipped or failed model is logged with the reason.

Validation: expanding-window walk-forward, refit every 4 weeks, first origin after 52 weeks of data. Hyperparameters tuned by walk-forward on origins up to 2022-12-31 with an equal budget (20 random configurations per tunable model). **Final test: origins 2023-01-01 onward, hyperparameters frozen.** Also: rolling 104-week window; leave-one-division-out; leave-hospital-out (grouped); leave-one-smoke-episode-out (retrospective, labelled as such); equal-sized training samples for high- and low-experience groups. All scaling, imputation and feature selection are fit on training data only.

Metrics: RMSE, MAE, MSE, MASE, bias, R²_OOS = 1 − MSPE_model / MSPE_benchmark against (i) persistence and (ii) the pooled AR model (the strong benchmark); conformal/quantile interval coverage and width; top-decile congestion detection (precision, recall, F1, AUROC, PR-AUC).
Regimes: normal, smoke (W4 = 1 in target week), post-smoke (4 weeks after); by E1 tercile; by capacity; eastern 2023 episode; held-out divisions.

Tests: Clark–West only for nested linear models estimated by OLS; Diebold–Mariano-type tests on panel loss differentials with state-and-week clustered variance for all others; block bootstrap by week for R²_OOS intervals. H6/H7: regression of the loss differential on smoke regime, E1 and their interaction, clustered by state and week — group differences are tested directly, not inferred from separate significance.

## 11. Causal ML

Partially linear DML (cross-fitting with folds grouped by state) for β; causal forest / GRF for heterogeneity by E1, capacity, metro, baseline occupancy, income, share 65+. Reported as causal only under the same identification conditions as §7; feature importance is never described as a causal effect.

## 12. Decision framework

Pass/fail criteria D1–D10, E1–E10, M1–M10, L1–L8 and the ranking weights (novelty 25, data 20, econometric credibility 20, economic significance 15, ML 15, journal 5) are those in the project brief. Integrity gates (data linkage, leakage, outcome validity, causal claims) are applied before scoring. Statistical significance earns no points by itself.

## 13. Known constraints stated in advance

- The US facility data cover 2020–2024 only, so every week is a pandemic week; pandemic confounding cannot be removed by dropping years, only reduced by fixed effects and COVID-burden controls.
- The facility data have no cause-specific ED visits, no waiting times, no discharges and no staffing. H1 (respiratory), boarding, LWBS, discharges and readmissions are NOT TESTABLE in US-W.
- Computer: 8 GB RAM, 4 cores. Heavy models are run on subsamples where needed and that is reported.
