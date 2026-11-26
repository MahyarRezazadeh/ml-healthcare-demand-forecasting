# Model card: one-week-ahead hospital occupancy forecast

## Model

Gradient-boosted trees (LightGBM) fitted to the one-week change in a hospital's inpatient occupancy rate, pooled across hospitals. Inputs are feature set A: the hospital's own recent occupancy (current value, lags, rolling means and standard deviation, same week last year, long-run mean), week-of-year harmonics, location, COVID share of occupied beds and log staffed beds. Hyperparameters are in `results/forecasting/tuned_hyperparameters.json`.

XGBoost with the same inputs performs the same (RMSE 5.447 against 5.449).

## Intended use

- A benchmark and starting point for weekly capacity planning forecasts at hospital level.
- A reference evaluation design for testing whether a new data source (here wildfire smoke) adds forecasting value.

## Not intended for

- Operational use in any hospital without re-estimation on that system's own data.
- Decisions about individual patients.
- Daily or hourly planning; the model is weekly.
- Triggering high-occupancy alerts as it stands (see "Alerts" below).

## Data

US HHS facility-level weekly capacity reports, August 2020 to April 2024. Trained on a fixed random sample of 1,000 short-term hospitals with an expanding window; tested on target weeks January 2023 to April 2024 (68,237 hospital-weeks).

## Performance in the test period

| Metric | LightGBM (A) | Pooled autoregression | No-change forecast |
|---|---|---|---|
| RMSE (percentage points) | 5.45 | 5.62 | 6.12 |
| MAE (percentage points) | 3.61 | 3.75 | 3.99 |
| Squared-error reduction vs no-change | 20.6% (95% interval 18.7% to 22.7%) | 15.5% | – |
| 90% interval coverage (feature set C) | 91.4% | 90.8% | 90.3% |

By subgroup (`results/forecasting/metrics_occ_h1.csv`, `extra_lodo.csv`, `extra_lho.csv`):

- Severe-smoke weeks: 19.8% reduction, similar to other weeks.
- Hospitals not seen in training: 19.4%.
- Census divisions not seen in training: 13.8% (Pacific) to 23.4% (West North Central).
- Hospitals in counties with more smoke history are forecast more accurately by every model, including the no-change forecast. This is a difference between hospitals, not an effect of smoke.

## Alerts

For weeks above a hospital's own 90th percentile, the model ranks well (AUROC 0.93) but at the natural threshold it finds 55% of such weeks, against 73% for the no-change forecast. A model trained for squared error shrinks toward the mean. An alerting use needs a classifier or a recalibrated threshold.

## Limitations

- All training and test weeks fall in the pandemic reporting period. Accuracy in a different period is not established.
- The target is the weekly average occupancy rate. Peaks within a week are not modelled.
- The sample is 1,000 hospitals because of an 8 GB memory limit. Hyperparameters were tuned with 8 random configurations on one chronological split.
- Occupancy is a ratio; changes in staffed beds move it without a change in patients. Log staffed beds is an input, but bed changes are not forecast.
- No patient-level or demographic information is used, so performance by patient group cannot be assessed. Performance by hospital location and size is reported in the metrics files.

## What was tested and did not help

Weather, current wildfire smoke, smoke realised in the target week, and county smoke history. See [results.md](results.md), section 2.
