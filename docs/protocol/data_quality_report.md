---
title: 'Data quality report'
date: '25 November 2026'
geometry: margin=2.3cm
fontsize: 10pt
---

# US hospital-week panel (HHS facility file)

- Raw file: 1,045,406 rows, 5,172 hospitals, 226 weeks (2019-12-29 to 2024-04-21); weeks start on Sunday. No duplicate hospital-weeks.
- Cleaned panel: 863,016 rows, 4,585 short-term and critical-access hospitals in the contiguous US, 195 weeks.
- Suppression: values 1–3 are coded −999999 in the source and set to missing (85,886 occupied-bed cells, 122,018 ICU cells in the raw file).
- Hospitals reporting at least 80% of weeks: 3,701 for occupancy, 2,313 for ED visits, 2,045 for ICU occupancy.
- Reporting break: the ED-visit field is optional from 11 June 2023.

Share of hospital-weeks with a reported value, by quarter:

| qtr    |   occ |   icu_occ |    ed |   ed_noncovid |
|:-------|------:|----------:|------:|--------------:|
| 2020Q3 | 0.854 |     0.476 | 0.907 |         0.787 |
| 2020Q4 | 0.912 |     0.541 | 0.994 |         0.925 |
| 2021Q1 | 0.899 |     0.533 | 0.997 |         0.915 |
| 2021Q2 | 0.896 |     0.519 | 0.998 |         0.902 |
| 2021Q3 | 0.910 |     0.547 | 0.998 |         0.934 |
| 2021Q4 | 0.916 |     0.543 | 0.998 |         0.937 |
| 2022Q1 | 0.912 |     0.538 | 0.998 |         0.926 |
| 2022Q2 | 0.895 |     0.507 | 0.998 |         0.905 |
| 2022Q3 | 0.896 |     0.511 | 0.998 |         0.924 |
| 2022Q4 | 0.908 |     0.521 | 0.998 |         0.911 |
| 2023Q1 | 0.910 |     0.522 | 0.998 |         0.899 |
| 2023Q2 | 0.899 |     0.505 | 0.905 |         0.788 |
| 2023Q3 | 0.893 |     0.501 | 0.508 |         0.450 |
| 2023Q4 | 0.901 |     0.511 | 0.470 |         0.426 |
| 2024Q1 | 0.906 |     0.515 | 0.461 |         0.413 |
| 2024Q2 | 0.900 |     0.504 | 0.455 |         0.391 |

# Smoke exposure

- NOAA HMS: 19 of about 6,870 daily files missing at source and 14 unreadable, all before 2016. Density classes are available from 2010.
- EPA AQS: 1,535 monitoring sites, 2.93 million site-days 2017–2024. A monitor-month background needs at least 15 non-plume days in the previous three years; 93.7% of site-days have one.
- Validation against the Stanford county product, 2020: Pearson 0.934, Spearman 0.763; 10,047.0 hospital-days at or above 20 µg/m³ in this measure against 9,312.0 in Stanford's, 8,186.0 in both.
- Smoke PM2.5 is available for 90.2% of hospital-weeks; 226 hospitals have no monitor within 100 km and enter only the plume-days models.
- HMS plumes describe smoke in the air column, not at the surface; the PM2.5 measure corrects for this only where monitors exist.

# Quebec ED-day panel

- Source: the MSSS hourly emergency-room file as archived every hour in a public third-party git repository; MSSS publishes no history. 143 installation codes, 115 with coordinates and continuous reporting.
- Days with fewer than 12 hourly snapshots are dropped. Patients present, patients waiting and length of stay start in February 2023.
- NAPS PM2.5: 2019, 2022 and 2023 files; 47 Quebec stations. Background uses non-plume days from the other available years, which is weaker than the three-year rule used for the US.
- One smoke season: 775 severe ED-days on 44 dates.

# Other

- County covariates: SAIPE 2019, Census population estimates 2019, CDC PLACES (2023 release, so not strictly predetermined).
- Census API could not be used without a key.
