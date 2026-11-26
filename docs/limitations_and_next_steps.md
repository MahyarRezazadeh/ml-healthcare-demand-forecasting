# Limitations and next steps

## What this project does not show

1. **It does not show that smoke never strains a hospital.** The outcome is weekly average occupancy at facility level. Congestion that lasts a few days, or that shows up as waiting time and not as occupied beds, would not be detected. The public file has no waiting times, discharges or cause-specific visits.
2. **It does not separate patient behaviour from hospital response.** Smoke may raise respiratory visits and lower other visits at the same time. With total load only, the two cannot be told apart.
3. **It does not cover a normal period.** Every week is between August 2020 and April 2024.
4. **It was not estimated on Canadian hospital data,** apart from one exploratory Quebec series covering a single smoke season.
5. **The forecasting sample is 1,000 hospitals.** The full sample did not fit in memory on the machine used.

## Deviations from the written protocol

Listed in [protocol/protocol_amendments.md](protocol/protocol_amendments.md). The main ones:

- 1,000-hospital forecasting sample; 8 tuning configurations per learner on one split, not 20 with walk-forward tuning.
- Lasso and elastic net refit every 13 weeks; slower learners every 26 weeks on at most 100,000 training rows.
- Neural and Prophet models on a single chronological split.
- No quantile models; intervals are split-conformal.
- Permutation placebo with 200 draws, not 500.

## Not done

- Forecasts on the full hospital sample.
- A wind-direction instrument for smoke exposure.
- Synthetic-control or matrix-completion estimates.
- Wild-cluster bootstrap inference.
- An end-to-end re-run of the pipeline in this repository layout (see "Verification status" in the README).

## Next steps that would add most

1. **Daily, cause-specific data.** Emergency visits by cause and time to physician would allow a test of respiratory demand and of waiting times, which is where a smoke effect is most likely.
2. **A classification model for high-occupancy weeks,** evaluated on recall at a fixed false-alarm rate, since the regression model is not suited to alerts.
3. **Probabilistic forecasts** (quantile gradient boosting) in place of fixed-width conformal intervals, so interval width varies with the hospital and the week.
4. **Hierarchical reconciliation** of hospital forecasts with regional totals.
5. **Monitoring.** A scheduled check of forecast error against the autoregressive benchmark, with an alert when the gain falls below a set level, would be the first requirement for production use.
