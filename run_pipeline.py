"""Run the pipeline stage by stage.

    python run_pipeline.py --list                 show every stage and its commands
    python run_pipeline.py --stage build          run one stage
    python run_pipeline.py --stage forecast --dry-run
    python run_pipeline.py --all                  run every core stage in order

Each command is a script in src/. The forecasting scripts are checkpointed: a finished model file is skipped,
so an interrupted stage can simply be started again.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
P = "data/processed"

STAGES = {
    "ingest": ("Download public raw data (about 3 GB) to data/raw or HDF_RAW_DIR", [
        "01_download_us.py all",
        "02_download_aux.py",
    ]),
    "build": ("Clean the hospital panel, build smoke exposure, assemble the hospital-week analysis panel", [
        "03_build_hhs_panel.py",
        "04_build_epa.py",
        "05_make_point_files.py",
        f"06_overlay_smoke_plumes.py {P}/pts_hosp.parquet {P}/hms_hosp.parquet 2010-01-01 2024-05-31",
        f"06_overlay_smoke_plumes.py {P}/pts_site.parquet {P}/hms_site.parquet 2017-01-01 2024-05-31",
        "07_build_covariates.py",
        "08_build_analysis_panel.py",
    ]),
    "forecast": ("Tune, run walk-forward forecasts, evaluate (1,000-hospital sample)", [
        "10_tune_hyperparameters.py",
        "11_forecast_walkforward.py occ 1 1",
        "11_forecast_walkforward.py occ 1 2",
        "13_forecast_validation_designs.py",
        "12_evaluate_forecasts.py occ 1",
        "11_forecast_walkforward.py occ 4 1",
        "12_evaluate_forecasts.py occ 4",
        "11_forecast_walkforward.py icu_occ 1 1",
        "12_evaluate_forecasts.py icu_occ 1",
        "11_forecast_walkforward.py l_ed 1 1",
        "12_evaluate_forecasts.py l_ed 1",
    ]),
    "causal": ("Panel regressions on the discovery and held-out samples, event study, double ML", [
        "20_panel_regressions.py discovery",
        "20_panel_regressions.py confirmation",
        "20_panel_regressions.py pooled",
        "22_event_study_placebo.py",
        "23_double_ml_causal_forest.py",
        "24_results_tables.py",
    ]),
    "report": ("Descriptive tables and figures", [
        "30_figures.py",
        "31_summary_figure.py",
    ]),
    # Secondary context. Its raw files are not fetched by the ingest stage; see docs/data.md.
    "quebec": ("Optional: Quebec emergency-department daily panel, regressions and forecasts", [
        "09_build_quebec_panel.py step1",
        f"06_overlay_smoke_plumes.py {P}/pts_qc_ed.parquet {P}/hms_qc_ed.parquet 2010-01-01 2023-12-31",
        f"06_overlay_smoke_plumes.py {P}/pts_naps.parquet {P}/hms_naps.parquet 2019-01-01 2023-12-31",
        "09_build_quebec_panel.py step2",
        "21_panel_regressions_quebec.py",
        "14_forecast_quebec_daily.py",
    ]),
}
CORE = ["ingest", "build", "forecast", "causal", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), action="append", help="stage to run (can be repeated)")
    ap.add_argument("--all", action="store_true", help="run the core stages: " + ", ".join(CORE))
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args()
    if a.list or not (a.stage or a.all):
        for name, (what, cmds) in STAGES.items():
            print(f"\n[{name}] {what}")
            for c in cmds:
                print(f"    python src/{c}")
        return
    for name in (CORE if a.all else a.stage):
        what, cmds = STAGES[name]
        print(f"\n=== {name}: {what}", flush=True)
        for c in cmds:
            parts = c.split()
            args = [sys.executable, os.path.join(SRC, parts[0])] + [os.path.join(ROOT, x) if x.startswith("data/") else x for x in parts[1:]]
            print("  $ python src/" + c, flush=True)
            if not a.dry_run:
                subprocess.run(args, cwd=SRC, check=True)


if __name__ == "__main__":
    main()
