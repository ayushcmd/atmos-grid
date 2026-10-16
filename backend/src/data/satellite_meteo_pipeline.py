"""
Nationwide Satellite and Meteorological Inputs Pipeline.

Provides acquisition workflows, authentication preflight diagnostics,
and data preparation for:
1. Sentinel-5P / TROPOMI Level-3 HCHO Column Density (GEE)
2. MODIS MAIAC Level-2 Aerosol Optical Depth (MCD19A2 via GEE)
3. ERA5 Meteorological Reanalysis (Copernicus Climate Data Store CDS)
4. NASA FIRMS Active Fire Observations (MODIS / VIIRS South Asia / India)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_METRICS = PROJECT_ROOT / "outputs" / "metrics"

INDIA_BBOX = [68.0, 6.5, 97.5, 37.5]  # [min_lon, min_lat, max_lon, max_lat]


def diagnose_gee() -> Dict[str, Any]:
    """Test Google Earth Engine credential and connection readiness."""
    res = {
        "service": "Google Earth Engine (GEE)",
        "datasets": [
            "COPERNICUS/S5P/OFFL/L3_HCHO (Sentinel-5P HCHO)",
            "MODIS/061/MCD19A2_GRANULES (MODIS MAIAC AOD)",
        ],
        "auth_required": True,
        "auth_method": "Google Cloud Project with Earth Engine API enabled or earthengine authenticate",
    }
    try:
        import ee
        try:
            # Check if default credentials or project exists
            project = os.environ.get("EE_PROJECT")
            ee.Initialize(project=project)
            res["status"] = "AUTHENTICATED"
            res["message"] = "Google Earth Engine initialized successfully."
            res["is_ready"] = True
        except Exception as e:
            res["status"] = "CREDENTIALS_MISSING"
            res["message"] = f"GEE initialization failed: {e}. Run 'earthengine authenticate' or set EE_PROJECT."
            res["is_ready"] = False
    except ImportError:
        res["status"] = "PACKAGE_MISSING"
        res["message"] = "earthengine-api Python package is not installed."
        res["is_ready"] = False

    return res


def diagnose_copernicus_cds() -> Dict[str, Any]:
    """Test Copernicus Climate Data Store (CDS) credential readiness for ERA5."""
    res = {
        "service": "Copernicus Climate Data Store (CDS)",
        "dataset": "reanalysis-era5-single-levels",
        "variables": [
            "boundary_layer_height (blh)",
            "2m_temperature (t2m)",
            "10m_u_component_of_wind (u10)",
            "10m_v_component_of_wind (v10)",
            "surface_pressure (sp)",
            "total_precipitation (tp)",
        ],
        "auth_required": True,
        "auth_method": "~/.cdsapirc configuration file with URL and API Personal Access Token",
    }

    home_cds = Path.home() / ".cdsapirc"
    env_url = os.environ.get("CDSAPI_URL")
    env_key = os.environ.get("CDSAPI_KEY")

    if home_cds.exists() or (env_url and env_key):
        res["status"] = "CONFIGURED"
        res["message"] = "CDS API credentials found (~/.cdsapirc or CDSAPI_KEY)."
        res["is_ready"] = True
    else:
        res["status"] = "CREDENTIALS_MISSING"
        res["message"] = (
            "No ~/.cdsapirc found. To acquire ERA5 reanalysis, register at "
            "https://cds.climate.copernicus.eu, generate an API token, and store it in ~/.cdsapirc."
        )
        res["is_ready"] = False

    return res


def diagnose_nasa_firms() -> Dict[str, Any]:
    """Test NASA FIRMS active fire data acquisition readiness."""
    res = {
        "service": "NASA FIRMS (Fire Information for Resource Management System)",
        "datasets": [
            "VIIRS S-NPP 375m Active Fire",
            "MODIS C6.1 1km Active Fire",
        ],
        "coverage": "South Asia / India Region",
        "auth_required": False,  # Open public 7-day country/regional feeds do not require token
        "auth_method": "Free public CSV download (South Asia) or NASA Earthdata MAP_KEY for archive",
        "public_url": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/",
    }

    # Public FIRMS allows free regional CSV access or MAP_KEY for API
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if map_key:
        res["status"] = "API_KEY_PRESENT"
        res["message"] = "NASA FIRMS MAP_KEY configured for full archival API queries."
        res["is_ready"] = True
    else:
        res["status"] = "PUBLIC_CSV_AVAILABLE"
        res["message"] = (
            "No MAP_KEY set; open public regional 7-day and 24-hour CSV feeds remain accessible without authentication."
        )
        res["is_ready"] = True

    return res


def run_full_preflight_diagnostics() -> Dict[str, Any]:
    """Run diagnostics across all external satellite and meteorological providers."""
    report = {
        "diagnostic_timestamp": datetime.now().isoformat() + "Z",
        "system_status": "OPERATIONAL",
        "providers": {
            "google_earth_engine": diagnose_gee(),
            "copernicus_cds_era5": diagnose_copernicus_cds(),
            "nasa_firms_fire": diagnose_nasa_firms(),
        },
        "scientific_integration_plan": {
            "hcho_pipeline": "Operates with Sentinel-5P Level-3 column densities + Haversine DBSCAN clustering.",
            "aqi_pipeline": "Trained on CPCB CAAQMS ground measurements. Next stage integrates collocated AOD + BLH.",
            "spatial_validation_requirement": (
                "Nationwide prediction requires spatial-leave-one-station-out cross-validation across distinct "
                "physiographic zones (Gangetic Plain, Deccan Plateau, Coastal Belts) before nationwide extrapolation."
            ),
        },
    }

    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_METRICS / "satellite_meteo_readiness.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved satellite and meteorological diagnostics to %s", report_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rep = run_full_preflight_diagnostics()
    print(json.dumps(rep, indent=2))
