# Data

All data are public. No raw or processed data are stored in this repository; the pipeline rebuilds them. Only result tables and a test-period sample of predictions are tracked.

## Sources

| Source | Content | Used for | Fetched by |
|---|---|---|---|
| US HHS, *COVID-19 Reported Patient Impact and Hospital Capacity by Facility* (healthdata.gov, dataset `anag-cw7u`) | Weekly beds, occupied beds, ICU beds, ED visits by hospital. 1,045,406 rows, 5,172 hospitals, weeks 2019-12-29 to 2024-04-21 | Targets | `01_download_us.py hhs` |
| NOAA Hazard Mapping System smoke polygons | Daily satellite-analyst smoke plumes with density class, 2005–2024 | Smoke exposure | `01_download_us.py hms` |
| US EPA Air Quality System, daily PM2.5 (parameters 88101 and 88502) | Monitor-day PM2.5 | Smoke exposure | `01_download_us.py epa` |
| Stanford smoke PM2.5 predictions, county-day, 2006–2020 (Harvard Dataverse, DJVMTV) | Modelled smoke PM2.5 | County smoke history; validation of the exposure measure | `01_download_us.py stanford` |
| NOAA nClimGrid-daily, county averages | Daily maximum temperature and precipitation | Weather controls and features | `02_download_aux.py` |
| FEMA disaster declarations (OpenFEMA API, incident type Fire) | County fire declarations | Excluding weeks with a local fire | `02_download_aux.py` |
| Census SAIPE 2019 (`est19all.xls`), Census population estimates 2019 by age and sex, CDC PLACES county file | Income, poverty, population, age shares, COPD prevalence | County characteristics (feature set D, robustness) | **manual**, see below |

### Files to place by hand

`02_download_aux.py` requests county characteristics from the Census ACS API, which now requires a key. The results therefore use three files that were downloaded by hand and are read by `07_build_covariates.py`:

```
data/raw/acs/est19all.xls          Census SAIPE 2019 state and county estimates
data/raw/acs/agesex/*.csv          Census county population estimates by age and sex, 2019 vintage (one file per state)
data/raw/acs/places_county.csv     CDC PLACES county data (2023 release)
```

### Quebec series (optional stage)

The `quebec` stage uses the official Quebec health ministry (MSSS) hourly emergency-department file as archived by a third-party public git repository (`jlomako/hospital-occupancy-tracker`), because the ministry publishes no history, and hourly PM2.5 from the Canadian National Air Pollution Surveillance (NAPS) programme. These files are not fetched by the `ingest` stage. `09_build_quebec_panel.py` expects them under `data/raw/canada/quebec/` and `data/raw/canada/naps/`. The series covers 115 emergency departments from October 2022 to December 2023 and is treated as exploratory.

## Cleaning rules (`03_build_hhs_panel.py`)

- Window: collection weeks 2020-08-02 to 2024-04-21 (195 weeks). Weeks start on Sunday.
- Hospitals: short-term and critical access hospitals in the contiguous United States, 4,585 hospitals. Forecasts and main regressions use short-term hospitals with at least 80% of weeks reported for the outcome.
- Suppressed cells (small counts coded −999999 in the source) are set to missing: 85,886 for inpatient beds used, 122,018 for ICU beds used.
- No duplicate hospital-weeks.
- ED visits: reported by nearly all hospitals until the week of 2023-06-04 and by about half afterwards, because the reporting requirement changed. ED forecasts are evaluated to 2023-06-10 only.

A full account is in [protocol/data_quality_report.md](protocol/data_quality_report.md).

## Smoke exposure (`04`–`08`)

1. Monitor-day smoke PM2.5 = PM2.5 above the monitor's background on days when a smoke plume is overhead, zero otherwise. The background is the monitor's median PM2.5 on plume-free days in the same calendar month over the previous three years.
2. Hospital-day exposure = inverse-distance-weighted average over monitors within 100 km.
3. Weekly measures at the hospital:

| Variable | Definition |
|---|---|
| `w1` | weekly mean smoke PM2.5 (µg/m³); missing if fewer than 5 days are available |
| `w2` | days under a heavy plume |
| `w3` | days under any plume |
| `w4` | severe smoke week: `w1 ≥ 15` or `w2 ≥ 2` |
| `w5` | weekly maximum of daily smoke PM2.5 |
| `E1` | county smoke history: standardised inverse hyperbolic sine of the number of days with smoke PM2.5 ≥ 20 µg/m³ in 2006–2019 |

Validation: against the Stanford county product in 2020, the Pearson correlation is 0.93 and agreement on days at or above 20 µg/m³ is 99.8%. Weekly exposure is available for 90.2% of hospital-weeks; 4,359 of 4,585 hospitals have a monitor within 100 km.

## Analysis panel

`data/processed/analysis_panel.parquet`: one row per hospital-week with targets, exposure, weather, county history and characteristics. Sample sizes in the main occupancy regression: 2,767 short-term hospitals and 521,439 hospital-weeks (334,740 to December 2022; 186,699 after).

## Tracked sample of predictions

`results/forecasting/sample_predictions_occ_h1.parquet`: test-period predictions for one-week-ahead inpatient occupancy, 1,000 hospitals, 68,237 rows.

| Column | Meaning |
|---|---|
| `hospital_pk` | HHS hospital identifier (public) |
| `week`, `target_week` | forecast origin and target week |
| `target` | realised occupancy (%) |
| `regime` | `smoke`, `post_smoke` or `normal` target week |
| `persistence`, `ar_AR`, `ridge_C`, `lgbm_A`, `lgbm_B`, `lgbm_C` | forecasts; the suffix is the feature set |
