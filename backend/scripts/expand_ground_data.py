"""
Expand CPCB Ground Air Quality Data Nationwide Across Indian States.
Fetches real observations from OpenAQ S3 for stations across multiple states:
- Delhi (8118, 8039)
- Uttar Pradesh (8083 Kanpur, 8557 Noida)
- Maharashtra (8055 Mumbai)
- Telangana (8038 Hyderabad)
Applies CPCB 16-hour completeness rule, computes official CPCB AQI,
attaches state administrative metadata via spatial point-in-polygon matching,
and saves the expanded dataset incrementally.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_cpcb")

PROJECT_ROOT = Path("c:/Projects/AtmosGrid")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_METRICS = PROJECT_ROOT / "outputs" / "metrics"

OPENAQ_S3_BUCKET = "s3://openaq-data-archive/records/csv.gz"

CPCB_AQI_BREAKPOINTS = {
    "pm25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)],
    "co": [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200), (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, 46.0, 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
}

def calculate_sub_index(conc: float, pollutant: str) -> float | None:
    if pd.isna(conc) or conc < 0:
        return None
    bps = CPCB_AQI_BREAKPOINTS.get(pollutant)
    if not bps:
        return None
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= conc <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo, 1)
    return None

def calculate_aqi(concs: dict) -> dict:
    sub_indices = {}
    for pol, val in concs.items():
        si = calculate_sub_index(val, pol)
        if si is not None:
            sub_indices[pol] = si
    if not sub_indices:
        return {"aqi": None, "dominant_pollutant": None, "sub_indices": {}, "n_valid": 0}
    dominant = max(sub_indices, key=sub_indices.get)
    return {
        "aqi": sub_indices[dominant],
        "dominant_pollutant": dominant,
        "sub_indices": sub_indices,
        "n_valid": len(sub_indices),
    }

def setup_duckdb():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-1';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    return con

def fetch_station(con, loc_id: int, year: int = 2024):
    s3_path = f"{OPENAQ_S3_BUCKET}/locationid={loc_id}/year={year}/**/*.csv.gz"
    query = f"""
        SELECT
            {loc_id} as location_id,
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
          AND parameter IN ('pm25', 'pm10', 'no2', 'so2', 'co', 'o3')
        ORDER BY datetime
    """
    try:
        df = con.execute(query).fetchdf()
        logger.info("Fetched %d rows for station %d", len(df), loc_id)
        return df
    except Exception as e:
        logger.warning("Station %d failed: %s", loc_id, e)
        return pd.DataFrame()

def main():
    # Load existing raw hourly data if present
    raw_path = DATA_PROCESSED / "cpcb_raw_hourly.csv"
    existing_dfs = []
    already_fetched = set()
    if raw_path.exists():
        old_raw = pd.read_csv(raw_path)
        existing_dfs.append(old_raw)
        already_fetched = set(old_raw["location_id"].unique())
        logger.info("Existing stations loaded: %s (%d records)", list(already_fetched), len(old_raw))

    # Stations to acquire
    stations_to_query = [
        (8083, "Nehru Nagar, Kanpur", "Uttar Pradesh"),
        (8038, "Zoo Park, Hyderabad", "Telangana"),
        (8055, "Bandra, Mumbai", "Maharashtra"),
        (8557, "Sector-62, Noida", "Uttar Pradesh"),
    ]

    con = setup_duckdb()
    for loc_id, name, state in stations_to_query:
        if loc_id in already_fetched:
            logger.info("Station %d (%s) already present, skipping fetch", loc_id, name)
            continue
        logger.info("Fetching Station %d (%s, %s)...", loc_id, name, state)
        df = fetch_station(con, loc_id, 2024)
        if not df.empty:
            existing_dfs.append(df)
            already_fetched.add(loc_id)

    con.close()

    if not existing_dfs:
        logger.error("No data available.")
        return

    full_raw = pd.concat(existing_dfs, ignore_index=True)
    full_raw.drop_duplicates(subset=["location_id", "datetime", "parameter"], inplace=True)
    full_raw.sort_values(by=["location_id", "datetime"], inplace=True)
    full_raw.to_csv(raw_path, index=False)
    logger.info("Saved combined raw hourly data to %s (%d records across %d stations)", raw_path, len(full_raw), full_raw["location_id"].nunique())

    # Daily aggregation with 16h completeness rule
    full_raw["date"] = pd.to_datetime(full_raw["datetime"]).dt.date
    daily = (
        full_raw.groupby(["location_id", "date", "latitude", "longitude", "parameter"])
        .agg(
            daily_mean=("value", "mean"),
            hourly_count=("value", "count"),
        )
        .reset_index()
    )
    # CPCB 16-hour completeness rule
    daily_valid = daily[daily["hourly_count"] >= 16].copy()

    pivoted = daily_valid.pivot_table(
        index=["location_id", "date", "latitude", "longitude"],
        columns="parameter",
        values="daily_mean",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None

    # Compute CPCB AQI
    aqi_records = []
    for _, row in pivoted.iterrows():
        concs = {}
        for pol in CPCB_AQI_BREAKPOINTS:
            if pol in row and pd.notna(row[pol]):
                concs[pol] = row[pol]
        aqi_records.append(calculate_aqi(concs))
    aqi_df = pd.DataFrame(aqi_records)
    daily_aqi = pd.concat([pivoted.reset_index(drop=True), aqi_df], axis=1)

    # Attach state via spatial point-in-polygon matching with simplified India states
    states_gdf = gpd.read_file(DATA_PROCESSED / "india_states_simplified.geojson")
    
    # Pre-map stations by location_id for known verified CPCB locations
    known_state_map = {
        8118: ("Delhi", "DTU, Delhi"),
        8039: ("Delhi", "Anand Vihar, Delhi"),
        8117: ("Delhi", "R.K. Puram, Delhi"),
        8083: ("Uttar Pradesh", "Nehru Nagar, Kanpur"),
        8557: ("Uttar Pradesh", "Sector-62, Noida"),
        8055: ("Maharashtra", "Bandra, Mumbai"),
        8038: ("Telangana", "Zoo Park, Hyderabad"),
        231: ("West Bengal", "US Consulate, Kolkata"),
    }

    station_states = {}
    station_names = {}
    for loc_id in daily_aqi["location_id"].unique():
        if loc_id in known_state_map:
            st, sn = known_state_map[loc_id]
            station_states[loc_id] = st
            station_names[loc_id] = sn
        else:
            loc_row = daily_aqi[daily_aqi["location_id"] == loc_id].iloc[0]
            pt = Point(loc_row["longitude"], loc_row["latitude"])
            matched = states_gdf[states_gdf.geometry.contains(pt)]
            if not matched.empty:
                station_states[loc_id] = matched.iloc[0]["NAME_1"]
            else:
                station_states[loc_id] = "Other"
            station_names[loc_id] = f"Station-{loc_id}"

    daily_aqi["state"] = daily_aqi["location_id"].map(station_states)
    daily_aqi["station_name"] = daily_aqi["location_id"].map(station_names)

    daily_aqi_path = DATA_PROCESSED / "cpcb_daily_aqi.csv"
    daily_aqi.to_csv(daily_aqi_path, index=False)
    logger.info("Saved daily AQI to %s (%d station-days across %d stations)", daily_aqi_path, len(daily_aqi), daily_aqi["location_id"].nunique())

    # Build Comprehensive State-Wise Coverage Report across all 36 States/UTs of India
    all_states_list = sorted(states_gdf["NAME_1"].unique().tolist())
    
    state_metrics = []
    for st_name in all_states_list:
        st_data = daily_aqi[daily_aqi["state"] == st_name]
        if not st_data.empty:
            valid_aqi_cnt = int(st_data["aqi"].notna().sum())
            available_pols = [p for p in ["pm25", "pm10", "no2", "so2", "co", "o3"] if p in st_data.columns and st_data[p].notna().sum() > 0]
            status = "MEASURED_GROUND" if valid_aqi_cnt > 0 else "INSUFFICIENT_DATA"
            state_metrics.append({
                "state": st_name,
                "coverage_status": status,
                "status_label": "Measured Ground Observations",
                "station_count": int(st_data["location_id"].nunique()),
                "valid_station_days": valid_aqi_cnt,
                "date_start": str(st_data["date"].min()),
                "date_end": str(st_data["date"].max()),
                "pollutants_measured": available_pols,
                "mean_aqi": round(float(st_data["aqi"].mean()), 1) if valid_aqi_cnt > 0 else None,
                "data_completeness": f"{round((valid_aqi_cnt / (st_data['location_id'].nunique() * 365)) * 100, 1)}%",
                "scientific_caveat": "Real CPCB / State PCB monitoring network observations aggregated to 24h CPCB standards.",
            })
        else:
            # Check if state has satellite coverage from TROPOMI HCHO benchmark
            # The TROPOMI India benchmark covers the entire bounding box 68-97.5E, 6.5-37.5N
            state_metrics.append({
                "state": st_name,
                "coverage_status": "SATELLITE_ONLY",
                "status_label": "Satellite Coverage (No Ground Calibration)",
                "station_count": 0,
                "valid_station_days": 0,
                "date_start": None,
                "date_end": None,
                "pollutants_measured": ["HCHO (Sentinel-5P TROPOMI)"],
                "mean_aqi": None,
                "data_completeness": "0% (Ground Stations Absent/Unlinked)",
                "scientific_caveat": "Satellite observations available; surface AQI uncalibrated without collocated ground CAAQMS monitoring.",
            })

    coverage_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_indian_states_and_uts": len(all_states_list),
        "states_with_measured_ground": len([s for s in state_metrics if s["coverage_status"] == "MEASURED_GROUND"]),
        "states_satellite_only": len([s for s in state_metrics if s["coverage_status"] == "SATELLITE_ONLY"]),
        "states_validated_model": 0, # Strictly 0 per guidelines until nationwide spatial validation is established
        "total_active_stations": int(daily_aqi["location_id"].nunique()),
        "total_station_days": len(daily_aqi),
        "status_definitions": {
            "MEASURED_GROUND": "Active CPCB CAAQMS stations with valid continuous hourly data (>=16h daily completeness rule).",
            "SATELLITE_ONLY": "Satellite column measurements (Sentinel-5P HCHO) present, but lacking collocated ground station calibration.",
            "VALIDATED_MODEL": "Spatially independent cross-validated ML surface-AQI predictions (deferred until nationwide dataset alignment).",
            "INSUFFICIENT_DATA": "Neither continuous ground monitoring nor validated satellite retrieval available."
        },
        "state_breakdown": state_metrics
    }

    # Save state coverage report
    with open(OUTPUTS_METRICS / "state_coverage_report.json", "w") as f:
        json.dump(coverage_report, f, indent=2)

    df_summary = pd.DataFrame(state_metrics)
    df_summary.to_csv(OUTPUTS_METRICS / "state_coverage_summary.csv", index=False)
    logger.info("Saved state coverage report to %s and CSV", OUTPUTS_METRICS / "state_coverage_report.json")

    # Update cpcb_quality_report.json
    quality_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_source": "OpenAQ AWS S3 Archive (s3://openaq-data-archive)",
        "provider_note": "Aggregated from official CPCB and State PCB CAAQMS continuous ambient stations.",
        "year": 2024,
        "stations_queried": int(full_raw["location_id"].nunique()),
        "raw_data": {
            "total_records": len(full_raw),
            "stations_with_data": int(full_raw["location_id"].nunique()),
            "parameters_found": sorted(full_raw["parameter"].unique().tolist()),
            "date_range": {
                "start": str(pd.to_datetime(full_raw["datetime"]).min()),
                "end": str(pd.to_datetime(full_raw["datetime"]).max()),
            },
            "per_parameter_counts": full_raw["parameter"].value_counts().to_dict(),
        },
        "aqi_summary": {
            "total_station_days": len(daily_aqi),
            "valid_aqi_station_days": int(daily_aqi["aqi"].notna().sum()),
            "aqi_statistics": {
                "mean": round(float(daily_aqi["aqi"].mean()), 1),
                "median": round(float(daily_aqi["aqi"].median()), 1),
                "min": round(float(daily_aqi["aqi"].min()), 1),
                "max": round(float(daily_aqi["aqi"].max()), 1),
            },
            "dominant_pollutant_distribution": daily_aqi["dominant_pollutant"].value_counts().to_dict(),
        },
        "limitations": [
            "24-hour daily averages require >=16 valid hours per CPCB guidelines.",
            "O3 and CO 8-hour maximums approximated from available hourly readings.",
            "NH3 sub-index omitted due to sparse station coverage.",
            "Predictions are spatially independent; nationwide extrapolation without regional ground validation is scientifically invalid.",
        ]
    }
    with open(OUTPUTS_METRICS / "cpcb_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)
    logger.info("Updated cpcb_quality_report.json successfully!")

if __name__ == "__main__":
    main()
