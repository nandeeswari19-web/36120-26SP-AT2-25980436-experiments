"""
Pull hourly historical weather data for Sydney from the Open-Meteo
Historical Weather API, save it as a CSV, then run it through the
package's aggregate_daily / add_cci / add_whc functions as a sanity
check.

Usage:
    python fetch_weather.py

Requires: requests, pandas, and nba-draft-features>=0.3.0 installed
(pip install -i https://test.pypi.org/simple/ nba-draft-features).

AT2 spec: use data prior to 2026 for train/val/test. 2026+ is reserved
as "production" data and must not be used for model development.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------
# Config — Sydney, per the AT2 brief
# ---------------------------------------------------------------------

LATITUDE = -33.8688
LONGITUDE = 151.2093

# Open-Meteo's ERA5 reanalysis archive starts 1940, but pick a
# reasonable modern window; adjust START_DATE if you want more history.
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"  # strictly pre-2026, per the spec

# All variables needed across both CCI and WHC formulas
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "snowfall",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path("data/raw")
RAW_HOURLY_PATH = OUTPUT_DIR / "sydney_hourly_raw.csv"


def fetch_historical_weather() -> pd.DataFrame:
    """Call the Open-Meteo archive API and return hourly data as a DataFrame."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Australia/Sydney",
    }

    print(f"Requesting {START_DATE} to {END_DATE} for ({LATITUDE}, {LONGITUDE})...")
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()

    payload = response.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Unexpected API response, no 'hourly' key: {payload}")

    df = pd.DataFrame(payload["hourly"])
    print(f"Retrieved {len(df):,} hourly rows.")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hourly = fetch_historical_weather()
    hourly.to_csv(RAW_HOURLY_PATH, index=False)
    print(f"Saved raw hourly data to {RAW_HOURLY_PATH}")

    # Quick sanity check by importing the package we just published.
    try:
        from nba_draft_features.aggregation import aggregate_daily
        from nba_draft_features.targets import add_cci, add_whc
    except ImportError:
        print(
            "\nnba-draft-features not installed in this environment — "
            "run:\n  pip install -i https://test.pypi.org/simple/ nba-draft-features\n"
            "then re-run this script to see aggregated CCI/WHC output.",
            file=sys.stderr,
        )
        return

    daily = aggregate_daily(hourly, datetime_col="time")
    daily = add_cci(daily)
    daily = add_whc(daily)

    daily_path = OUTPUT_DIR.parent / "processed" / "sydney_daily_targets.csv"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False)

    print(f"\nSaved daily aggregated + target data to {daily_path}")
    print(f"Rows: {len(daily):,}  |  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print("\nFirst 5 rows of CCI/WHC output:")
    print(daily[["date", "cci", "whi", "whc_label"]].head())
    print("\nCCI summary stats:")
    print(daily["cci"].describe())
    print("\nWHC class distribution:")
    print(daily["whc_label"].value_counts())


if __name__ == "__main__":
    main()
