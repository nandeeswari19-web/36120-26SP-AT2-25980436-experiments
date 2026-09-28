"""
Run this from inside the EXPERIMENTS repo (36120-26SP-AT2-25980436-experiments),
AFTER running both notebooks (comfort_climate and weather_hazard), to export:

  1. The trained models -> models/comfort_climate/cci_model.joblib
                            models/weather_hazard/whc_model.joblib
  2. Feature lookup tables the API needs to serve predictions for any
     historical date -> models/comfort_climate/cci_feature_lookup.csv
                        models/weather_hazard/whc_feature_lookup.csv

Copy all 4 output files into the API repo's models/ folder afterwards.

NOTE: this script re-derives daily_clean the same way both notebooks do.
If you changed anything in the notebooks (feature engineering, target
definition), update this script to match before running it, or the API's
predictions won't line up with what the notebooks actually trained on.
"""

import joblib
import pandas as pd

from nba_draft_features.aggregation import aggregate_daily
from nba_draft_features.targets import add_cci, add_whc, shift_target

daily = pd.read_csv("data/processed/sydney_daily_targets.csv", parse_dates=["date"])

# ---------------------------------------------------------------------
# CCI feature lookup
# ---------------------------------------------------------------------
cci_daily = add_cci(daily.copy())
cci_daily["cci_target_3d"] = shift_target(cci_daily, "cci", horizon_days=3, date_col="date")
cci_clean = cci_daily.dropna(subset=["cci_target_3d"]).reset_index(drop=True)

cci_clean["temp_lag1"] = cci_clean["temperature_2m"].shift(1)
cci_clean["temp_roll7"] = cci_clean["temperature_2m"].rolling(7).mean()
cci_clean["cci_lag1"] = cci_clean["cci"].shift(1)
cci_clean["month"] = cci_clean["date"].dt.month
cci_clean["day_of_year"] = cci_clean["date"].dt.dayofyear
cci_clean = cci_clean.dropna().reset_index(drop=True)

CCI_FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover",
    "precipitation", "temp_lag1", "temp_roll7", "cci_lag1", "month", "day_of_year",
]
cci_lookup = cci_clean[["date"] + CCI_FEATURE_COLS]
cci_lookup.to_csv("models/comfort_climate/cci_feature_lookup.csv", index=False)
print("Saved models/comfort_climate/cci_feature_lookup.csv:", cci_lookup.shape)

# ---------------------------------------------------------------------
# WHC feature lookup
# ---------------------------------------------------------------------
whc_daily = add_whc(daily.copy())
whc_daily["whc_target_7d"] = shift_target(whc_daily, "whc_class", horizon_days=7, date_col="date")
whc_clean = whc_daily.dropna(subset=["whc_target_7d"]).reset_index(drop=True)

whc_clean["wind_gusts_lag1"] = whc_clean["wind_gusts_10m"].shift(1)
whc_clean["precip_roll3"] = whc_clean["precipitation"].rolling(3).sum()
whc_clean["whi_lag1"] = whc_clean["whi"].shift(1)
whc_clean["month"] = whc_clean["date"].dt.month
whc_clean["day_of_year"] = whc_clean["date"].dt.dayofyear
whc_clean = whc_clean.dropna().reset_index(drop=True)

WHC_FEATURE_COLS = [
    "precipitation", "wind_gusts_10m", "cloud_cover", "snowfall", "temperature_2m",
    "wind_gusts_lag1", "precip_roll3", "whi_lag1", "month", "day_of_year",
]
whc_lookup = whc_clean[["date"] + WHC_FEATURE_COLS]
whc_lookup.to_csv("models/weather_hazard/whc_feature_lookup.csv", index=False)
print("Saved models/weather_hazard/whc_feature_lookup.csv:", whc_lookup.shape)

print("\nDone. Copy these 4 files into the API repo's models/ folder:")
print("  models/comfort_climate/cci_model.joblib")
print("  models/comfort_climate/cci_feature_lookup.csv")
print("  models/weather_hazard/whc_model.joblib")
print("  models/weather_hazard/whc_feature_lookup.csv")
