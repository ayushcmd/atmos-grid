#!/usr/bin/env python
"""
CPCB Ground Air Quality Data Acquisition Pipeline Entry Point.

Downloads raw pollutant observations from OpenAQ S3 archive for known
Indian CPCB monitoring stations, computes daily AQI using CPCB breakpoints,
and generates a data quality report.

Usage:
    .venv\\Scripts\\python backend\\scripts\\run_cpcb_acquisition.py

The script uses a curated list of known Indian CPCB station IDs from OpenAQ.
To discover station IDs programmatically, use the OpenAQ API v3.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.data.fetch_cpcb_openaq import run_acquisition

# Curated list of known Indian CPCB/SPCB station location IDs in OpenAQ.
# These were identified via the OpenAQ API for Indian stations with good coverage.
# Each tuple: (location_id, description)
KNOWN_INDIA_STATIONS = [
    (8118, "DTU Delhi"),
    (8039, "Anand Vihar Delhi"),
    (8117, "R.K. Puram Delhi"),
    (2316, "US Embassy Delhi"),
    (1485, "US Consulate Mumbai"),
    (8557, "Sector-62 Noida"),
    (8559, "Sector-125 Noida"),
    (8083, "Nehru Nagar Kanpur"),
    (8167, "Lalbagh Lucknow"),
    (8084, "Talkatora Lucknow"),
    (8034, "Ghodasar Ahmedabad"),
    (8055, "Bandra Mumbai"),
    (8570, "Karve Road Pune"),
    (8038, "Zoo Park Hyderabad"),
    (8562, "Mandir Marg Delhi"),
    (231, "Kolkata"),
    (8560, "NSIT Dwarka Delhi"),
    (8085, "Central School Lucknow"),
    (8037, "ITO Delhi"),
    (8040, "Pusa Delhi"),
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_cpcb_acquisition")

    import argparse
    parser = argparse.ArgumentParser(description="Run CPCB data acquisition from OpenAQ S3")
    parser.add_argument("--max-stations", type=int, default=3, help="Max stations to acquire (default: 3)")
    parser.add_argument("--year", type=int, default=2024, help="Year to acquire (default: 2024)")
    args = parser.parse_args()

    selected_stations = KNOWN_INDIA_STATIONS[:args.max_stations]
    station_ids = [s[0] for s in selected_stations]
    logger.info("Starting CPCB data acquisition for %d stations...", len(station_ids))
    logger.info("Stations: %s", [s[1] for s in selected_stations])

    try:
        raw_df, daily_aqi_df, quality_report = run_acquisition(
            station_ids=station_ids,
            year=args.year,
            parameters=["pm25", "pm10", "no2", "so2", "co", "o3"],
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)

    # Report results
    logger.info("=" * 60)
    logger.info("CPCB Data Acquisition Complete")
    logger.info("=" * 60)

    status = quality_report.get("status", "OK")
    if status == "NO_DATA":
        logger.warning("No data retrieved. Check station IDs and S3 connectivity.")
        logger.warning("Message: %s", quality_report.get("message", ""))
        sys.exit(1)

    raw_info = quality_report.get("raw_data", {})
    logger.info("Total raw records: %s", raw_info.get("total_records", 0))
    logger.info("Stations with data: %s", raw_info.get("stations_with_data", 0))
    logger.info("Parameters found: %s", raw_info.get("parameters_found", []))
    logger.info(
        "Date range: %s to %s",
        raw_info.get("date_range", {}).get("start", "?"),
        raw_info.get("date_range", {}).get("end", "?"),
    )

    aqi_info = quality_report.get("aqi_summary", {})
    if aqi_info:
        logger.info("Valid AQI station-days: %s", aqi_info.get("valid_aqi_station_days", 0))
        aqi_stats = aqi_info.get("aqi_statistics", {})
        logger.info(
            "AQI stats — Mean: %s, Median: %s, Min: %s, Max: %s",
            aqi_stats.get("mean"), aqi_stats.get("median"),
            aqi_stats.get("min"), aqi_stats.get("max"),
        )

    logger.info("Outputs saved to:")
    logger.info("  data/processed/cpcb_raw_hourly.csv")
    logger.info("  data/processed/cpcb_daily_aqi.csv")
    logger.info("  outputs/metrics/cpcb_quality_report.json")


if __name__ == "__main__":
    main()
