"""
CPCB Ground Air Quality Data Acquisition via OpenAQ AWS S3 Archive.

This module downloads Indian Central Pollution Control Board (CPCB) monitoring
station observations from the OpenAQ open data archive on AWS S3. The archive
is publicly accessible without authentication.

Data Source
-----------
- OpenAQ AWS S3 Archive: s3://openaq-data-archive/records/csv.gz/
- Provider: OpenAQ aggregates data from CPCB and State PCBs
- Stations: 500+ CAAQMS continuous monitoring stations across India
- Parameters: PM2.5, PM10, NO2, SO2, CO, O3, and meteorological variables
- Temporal Coverage: ~2015 to present, hourly resolution

CPCB AQI Methodology Note
--------------------------
OpenAQ provides raw pollutant concentrations, NOT official CPCB AQI values.
If AQI is needed, it must be calculated from sub-index formulas defined in:
  CPCB National Air Quality Index (2014)
  https://cpcb.nic.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvRmluYWxfUmVwb3J0X0FRSS5wZGY=

The CPCB AQI uses 24-hour averages for PM2.5, PM10, SO2, NO2, NH3, CO
and 8-hour maximum for O3. Sub-indices are calculated per pollutant using
breakpoint tables, and the overall AQI is the maximum sub-index.

Limitations:
- OpenAQ may not mirror all CPCB stations
- Data gaps are common due to instrument maintenance
- Hourly values must be aggregated to 24h/8h before AQI calculation
- NH3 is required for complete CPCB AQI but may have poor coverage
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_METRICS = PROJECT_ROOT / "outputs" / "metrics"

# OpenAQ S3 configuration
OPENAQ_S3_BUCKET = "s3://openaq-data-archive/records/csv.gz"

# CPCB AQI breakpoint tables (24-hour averages except O3 which uses 8-hour max)
# Format: [(Clo, Chi, Ilo, Ihi), ...] where C=concentration, I=index
# Source: CPCB National Air Quality Index (2014), Table 2
CPCB_AQI_BREAKPOINTS = {
    "pm25": [  # µg/m³, 24-hour average
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "pm10": [  # µg/m³, 24-hour average
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "no2": [  # µg/m³, 24-hour average
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 800, 401, 500),
    ],
    "so2": [  # µg/m³, 24-hour average
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2400, 401, 500),
    ],
    "co": [  # mg/m³, 8-hour maximum
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 46.0, 401, 500),
    ],
    "o3": [  # µg/m³, 8-hour maximum
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}


def calculate_sub_index(concentration: float, pollutant: str) -> float | None:
    """Calculate CPCB AQI sub-index for a single pollutant.

    Parameters
    ----------
    concentration : float
        Pollutant concentration in appropriate units and averaging period.
    pollutant : str
        One of: pm25, pm10, no2, so2, co, o3.

    Returns
    -------
    float or None
        Sub-index value (0-500), or None if concentration is out of range
        or pollutant is unknown.
    """
    if pd.isna(concentration) or concentration < 0:
        return None

    breakpoints = CPCB_AQI_BREAKPOINTS.get(pollutant)
    if breakpoints is None:
        return None

    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            # Linear interpolation within breakpoint range
            sub_index = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
            return round(sub_index, 1)

    # Concentration exceeds highest breakpoint
    return None


def calculate_aqi(pollutant_concentrations: dict[str, float]) -> dict:
    """Calculate overall CPCB AQI from pollutant concentrations.

    The overall AQI is the maximum of the individual sub-indices.
    At least one valid sub-index is required.

    Parameters
    ----------
    pollutant_concentrations : dict
        Mapping of pollutant name to concentration value.
        Expected keys: pm25, pm10, no2, so2, co, o3 (all optional).

    Returns
    -------
    dict
        {
            'aqi': float,  # Overall AQI value
            'dominant_pollutant': str,  # Pollutant with highest sub-index
            'sub_indices': dict,  # Per-pollutant sub-indices
            'n_valid': int,  # Number of pollutants with valid sub-indices
        }
    """
    sub_indices = {}
    for pollutant, conc in pollutant_concentrations.items():
        si = calculate_sub_index(conc, pollutant)
        if si is not None:
            sub_indices[pollutant] = si

    if not sub_indices:
        return {
            "aqi": None,
            "dominant_pollutant": None,
            "sub_indices": {},
            "n_valid": 0,
        }

    dominant = max(sub_indices, key=sub_indices.get)
    return {
        "aqi": sub_indices[dominant],
        "dominant_pollutant": dominant,
        "sub_indices": sub_indices,
        "n_valid": len(sub_indices),
    }


def discover_india_stations(
    con: duckdb.DuckDBPyConnection,
    sample_location_ids: list[int] | None = None,
    year: int = 2024,
) -> pd.DataFrame:
    """Discover Indian CPCB monitoring stations in the OpenAQ archive.

    Queries the S3 archive to find distinct stations with their coordinates
    and available parameters.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with httpfs configured.
    sample_location_ids : list of int, optional
        If provided, only query these specific location IDs.
    year : int
        Year to search for station data.

    Returns
    -------
    pd.DataFrame
        Station metadata with columns: location_id, location_name,
        latitude, longitude, parameters (list of measured pollutants).
    """
    if sample_location_ids:
        # Query specific known station IDs
        all_stations = []
        for loc_id in sample_location_ids:
            s3_path = f"{OPENAQ_S3_BUCKET}/locationid={loc_id}/year={year}/**/*.csv.gz"
            try:
                query = f"""
                    SELECT DISTINCT
                        {loc_id} as location_id,
                        location as location_name,
                        lat as latitude,
                        lon as longitude,
                        parameter
                    FROM read_csv_auto('{s3_path}', filename=true)
                    LIMIT 100
                """
                df = con.execute(query).fetchdf()
                all_stations.append(df)
            except Exception as e:
                logger.warning("No data for location %d in %d: %s", loc_id, year, e)
        if all_stations:
            return pd.concat(all_stations, ignore_index=True)
        return pd.DataFrame()
    else:
        logger.info("Station discovery without IDs requires known location IDs.")
        logger.info("Use the OpenAQ API to get Indian station location IDs first.")
        return pd.DataFrame()


def fetch_station_data(
    con: duckdb.DuckDBPyConnection,
    location_id: int,
    year: int,
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch raw measurement data for a single station from OpenAQ S3.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with httpfs configured for anonymous S3 access.
    location_id : int
        OpenAQ location ID.
    year : int
        Year to fetch data for.
    parameters : list of str, optional
        If provided, filter to these parameter names (e.g., ['pm25', 'pm10']).

    Returns
    -------
    pd.DataFrame
        Raw measurements with columns: location_id, location_name, datetime,
        lat, lon, parameter, unit, value.
    """
    s3_path = f"{OPENAQ_S3_BUCKET}/locationid={location_id}/year={year}/**/*.csv.gz"

    param_filter = ""
    if parameters:
        param_list = ", ".join(f"'{p}'" for p in parameters)
        param_filter = f"AND parameter IN ({param_list})"

    query = f"""
        SELECT
            {location_id} as location_id,
            location as location_name,
            CAST(datetime AS TIMESTAMP) AS datetime,
            lat as latitude,
            lon as longitude,
            parameter,
            units as unit,
            value
        FROM read_csv_auto('{s3_path}', filename=true)
        WHERE value IS NOT NULL
          AND value >= 0
          AND value <= 2000
          {param_filter}
        ORDER BY datetime
    """
    try:
        df = con.execute(query).fetchdf()
        logger.info(
            "Fetched %d records for location %d (year %d)",
            len(df), location_id, year,
        )
        return df
    except Exception as e:
        logger.error("Failed to fetch location %d: %s", location_id, e)
        return pd.DataFrame()


def setup_duckdb_s3() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for anonymous S3 access.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Ready-to-use connection with httpfs loaded.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-1';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    return con


def aggregate_daily_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly measurements to daily averages per station per pollutant.

    For AQI calculation, CPCB requires:
    - 24-hour average for PM2.5, PM10, NO2, SO2, NH3
    - 8-hour running maximum for O3, CO

    This function computes 24-hour averages for all pollutants as a first
    approximation. The 8-hour max for O3 and CO requires sub-daily data
    and is noted as a limitation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw hourly measurements with columns: datetime, location_id,
        parameter, value, latitude, longitude.

    Returns
    -------
    pd.DataFrame
        Daily aggregated data with one row per station per date,
        pollutant columns pivoted.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

    # 24-hour mean aggregation (simplification — see docstring)
    daily = (
        df.groupby(["location_id", "date", "latitude", "longitude", "parameter"])
        .agg(
            daily_mean=("value", "mean"),
            hourly_count=("value", "count"),
        )
        .reset_index()
    )

    # Require at least 16 hours of data for a valid 24-hour average
    # (CPCB specifies 16-hour minimum for most pollutants)
    daily = daily[daily["hourly_count"] >= 16].copy()

    # Pivot pollutants to columns
    pivoted = daily.pivot_table(
        index=["location_id", "date", "latitude", "longitude"],
        columns="parameter",
        values="daily_mean",
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None
    return pivoted


def compute_daily_aqi(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute CPCB AQI for each station-day from daily pollutant averages.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Pivoted daily data with pollutant columns (pm25, pm10, etc.).

    Returns
    -------
    pd.DataFrame
        Input dataframe with added columns: aqi, dominant_pollutant, n_valid_pollutants.
    """
    if daily_df.empty:
        return daily_df

    aqi_results = []
    for _, row in daily_df.iterrows():
        concentrations = {}
        for pollutant in CPCB_AQI_BREAKPOINTS:
            if pollutant in row and pd.notna(row[pollutant]):
                concentrations[pollutant] = row[pollutant]

        result = calculate_aqi(concentrations)
        aqi_results.append(result)

    aqi_df = pd.DataFrame(aqi_results)
    result = pd.concat([daily_df.reset_index(drop=True), aqi_df], axis=1)
    return result


def generate_data_quality_report(
    raw_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    station_ids: list[int],
    year: int,
) -> dict:
    """Generate a data quality report for the acquired CPCB data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw hourly measurements.
    daily_df : pd.DataFrame
        Daily aggregated data with AQI.
    station_ids : list of int
        Station IDs that were queried.
    year : int
        Year of data.

    Returns
    -------
    dict
        Quality report with coverage, missing data, and summary statistics.
    """
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_source": "OpenAQ AWS S3 Archive (s3://openaq-data-archive)",
        "provider_note": (
            "OpenAQ aggregates data from CPCB and State PCB CAAQMS stations. "
            "Raw pollutant concentrations are provided, not official CPCB AQI. "
            "AQI is calculated here using CPCB 2014 breakpoint methodology."
        ),
        "year": year,
        "stations_queried": len(station_ids),
    }

    if raw_df.empty:
        report["status"] = "NO_DATA"
        report["message"] = "No records retrieved from OpenAQ S3 for the queried stations."
        return report

    report["raw_data"] = {
        "total_records": len(raw_df),
        "stations_with_data": int(raw_df["location_id"].nunique()),
        "parameters_found": sorted(raw_df["parameter"].unique().tolist()),
        "date_range": {
            "start": str(raw_df["datetime"].min()),
            "end": str(raw_df["datetime"].max()),
        },
        "per_parameter_counts": raw_df["parameter"].value_counts().to_dict(),
    }

    if not daily_df.empty and "aqi" in daily_df.columns:
        valid_aqi = daily_df[daily_df["aqi"].notna()]
        report["aqi_summary"] = {
            "total_station_days": len(daily_df),
            "valid_aqi_station_days": len(valid_aqi),
            "aqi_statistics": {
                "mean": round(float(valid_aqi["aqi"].mean()), 1) if len(valid_aqi) > 0 else None,
                "median": round(float(valid_aqi["aqi"].median()), 1) if len(valid_aqi) > 0 else None,
                "min": round(float(valid_aqi["aqi"].min()), 1) if len(valid_aqi) > 0 else None,
                "max": round(float(valid_aqi["aqi"].max()), 1) if len(valid_aqi) > 0 else None,
            },
        }
        if "dominant_pollutant" in valid_aqi.columns:
            report["aqi_summary"]["dominant_pollutant_distribution"] = (
                valid_aqi["dominant_pollutant"].value_counts().to_dict()
            )

    report["limitations"] = [
        "O3 and CO sub-indices should use 8-hour running maximum, not 24-hour average.",
        "NH3 sub-index is not computed due to limited NH3 data availability.",
        "Minimum 16 hours required for valid daily average (CPCB standard).",
        "OpenAQ station IDs may not map 1:1 to official CPCB station codes.",
    ]

    return report


def run_acquisition(
    station_ids: list[int],
    year: int = 2024,
    parameters: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the complete CPCB data acquisition and AQI calculation pipeline.

    Parameters
    ----------
    station_ids : list of int
        OpenAQ location IDs for Indian CPCB stations.
    year : int
        Year to acquire data for.
    parameters : list of str, optional
        Pollutant parameters to fetch. Defaults to AQI-relevant pollutants.

    Returns
    -------
    tuple of (raw_df, daily_aqi_df, quality_report)
    """
    if parameters is None:
        parameters = ["pm25", "pm10", "no2", "so2", "co", "o3"]

    logger.info("Setting up DuckDB S3 connection...")
    con = setup_duckdb_s3()

    logger.info("Fetching data for %d stations (year %d)...", len(station_ids), year)
    all_raw = []
    for loc_id in station_ids:
        df = fetch_station_data(con, loc_id, year, parameters)
        if not df.empty:
            all_raw.append(df)

    if not all_raw:
        logger.warning("No data retrieved for any station.")
        raw_df = pd.DataFrame()
        daily_aqi = pd.DataFrame()
    else:
        raw_df = pd.concat(all_raw, ignore_index=True)
        logger.info("Total raw records: %d", len(raw_df))

        # Aggregate to daily and compute AQI
        daily_df = aggregate_daily_pollutants(raw_df)
        daily_aqi = compute_daily_aqi(daily_df)
        logger.info("Daily AQI records: %d", len(daily_aqi))

    # Generate quality report
    quality_report = generate_data_quality_report(raw_df, daily_aqi, station_ids, year)

    # Save outputs
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)

    if not raw_df.empty:
        raw_path = DATA_PROCESSED / "cpcb_raw_hourly.csv"
        raw_df.to_csv(raw_path, index=False)
        logger.info("Raw data saved to %s", raw_path)

    if not daily_aqi.empty:
        aqi_path = DATA_PROCESSED / "cpcb_daily_aqi.csv"
        daily_aqi.to_csv(aqi_path, index=False)
        logger.info("Daily AQI saved to %s", aqi_path)

    report_path = OUTPUTS_METRICS / "cpcb_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=2, default=str)
    logger.info("Quality report saved to %s", report_path)

    con.close()
    return raw_df, daily_aqi, quality_report
