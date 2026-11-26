"""Shared configuration. Every script in src/ imports from here.

Paths are relative to the repository root. Raw downloads (about 3 GB) go to data/raw by default;
set the environment variable HDF_RAW_DIR to keep them somewhere else (for example outside a synced folder).
"""
import os
SEED = 20261125
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("HDF_RAW_DIR", os.path.join(PROJ, "data", "raw")).replace("\\", "/")
PROC = os.path.join(PROJ, "data", "processed")           # analysis panels (parquet); not tracked in git
DESC = os.path.join(PROJ, "results", "descriptive")
ECON = os.path.join(PROJ, "results", "causal")           # panel regressions, event study, double ML
ML = os.path.join(PROJ, "results", "forecasting")        # forecast metrics, tests, tuned hyperparameters
ROB = os.path.join(PROJ, "results", "robustness")
FIG = os.path.join(PROJ, "reports", "figures")
TAB = os.path.join(PROJ, "reports", "tables")
LOGS = os.path.join(PROJ, "logs")
for _d in (RAW, PROC, DESC, ECON, ML, ROB, FIG, TAB, LOGS):
    os.makedirs(_d, exist_ok=True)
WIN_START, WIN_END = "2020-07-31", "2024-04-26"      # HHS collection weeks (Fri start)
DISC_END = "2022-12-31"                               # discovery sample ends; confirmation after
EXP_PRE_END = "2019-12-31"                            # prior-experience window ends
SEVERE_PM = 20.0                                      # ug/m3 threshold for a severe smoke day
MONITOR_RADIUS_KM = 100.0
MIN_COVERAGE = 0.80

DIVISION = {  # census divisions
 **dict.fromkeys(["CT","ME","MA","NH","RI","VT"], "1_NewEngland"),
 **dict.fromkeys(["NJ","NY","PA"], "2_MidAtlantic"),
 **dict.fromkeys(["IL","IN","MI","OH","WI"], "3_ENCentral"),
 **dict.fromkeys(["IA","KS","MN","MO","NE","ND","SD"], "4_WNCentral"),
 **dict.fromkeys(["DE","DC","FL","GA","MD","NC","SC","VA","WV"], "5_SAtlantic"),
 **dict.fromkeys(["AL","KY","MS","TN"], "6_ESCentral"),
 **dict.fromkeys(["AR","LA","OK","TX"], "7_WSCentral"),
 **dict.fromkeys(["AZ","CO","ID","MT","NV","NM","UT","WY"], "8_Mountain"),
 **dict.fromkeys(["CA","OR","WA"], "9_Pacific"),
}
WEST = {"8_Mountain", "9_Pacific"}

