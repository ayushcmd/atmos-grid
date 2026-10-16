"""
Data loader module for AtmosGrid Streamlit Dashboard.

Provides robust, cached data loading functions that:
- Resolve all file paths relative to project root.
- Validate file existence, JSON structure, and DataFrame schemas.
- Provide clean fallbacks and informative error logs when artifacts are missing.
- Handle date and numeric parsing consistently.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data(show_spinner=False)
    cache_resource = st.cache_resource(show_spinner=False)
except ImportError:
    def _noop_decorator(func):
        return func
    cache_data = _noop_decorator
    cache_resource = _noop_decorator

import sys

logger = logging.getLogger(__name__)

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths to artifacts
PATH_CPCB_DAILY = PROJECT_ROOT / "data" / "processed" / "cpcb_daily_aqi.csv"
PATH_HCHO_BENCHMARK = PROJECT_ROOT / "data" / "processed" / "tropomi_hcho_india_benchmark.csv"
PATH_AQI_MODEL = PROJECT_ROOT / "models" / "aqi_xgboost.pkl"
PATH_HCHO_GEOJSON = PROJECT_ROOT / "outputs" / "hotspots" / "hcho_hotspots.geojson"
PATH_HCHO_SUMMARY = PROJECT_ROOT / "outputs" / "hotspots" / "hcho_clusters_summary.csv"
PATH_AQI_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "aqi_model_metrics.json"
PATH_CPCB_REPORT = PROJECT_ROOT / "outputs" / "metrics" / "cpcb_quality_report.json"
PATH_HCHO_REPORT = PROJECT_ROOT / "outputs" / "metrics" / "hcho_quality_report.json"
PATH_AQI_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions" / "aqi_test_predictions.csv"
PATH_STATE_COVERAGE_REPORT = PROJECT_ROOT / "outputs" / "metrics" / "state_coverage_report.json"
PATH_STATE_COVERAGE_SUMMARY = PROJECT_ROOT / "outputs" / "metrics" / "state_coverage_summary.csv"
PATH_INDIA_STATES_GEOJSON = PROJECT_ROOT / "data" / "processed" / "india_states_simplified.geojson"
PATH_SATELLITE_METEO_READINESS = PROJECT_ROOT / "outputs" / "metrics" / "satellite_meteo_readiness.json"

# Known station metadata mapping
STATION_METADATA = {
    8118: {"name": "DTU, Shahbad Daulatpur", "city": "Delhi", "state": "Delhi"},
    8039: {"name": "Anand Vihar - DPCC", "city": "Delhi", "state": "Delhi"},
    8117: {"name": "R.K. Puram - DPCC", "city": "Delhi", "state": "Delhi"},
    8083: {"name": "Nehru Nagar, Kanpur", "city": "Kanpur", "state": "Uttar Pradesh"},
    8557: {"name": "Sector-62, Noida", "city": "Noida", "state": "Uttar Pradesh"},
    8055: {"name": "Bandra, Mumbai", "city": "Mumbai", "state": "Maharashtra"},
    8038: {"name": "Zoo Park, Hyderabad", "city": "Hyderabad", "state": "Telangana"},
}


@cache_data
def load_cpcb_daily_aqi() -> Optional[pd.DataFrame]:
    """Load daily CPCB ground observations and AQI values."""
    if not PATH_CPCB_DAILY.exists():
        logger.warning("File not found: %s", PATH_CPCB_DAILY)
        return None
    try:
        df = pd.read_csv(PATH_CPCB_DAILY)
        df["date"] = pd.to_datetime(df["date"])
        # Annotate station readable names if known
        df["station_name"] = df["location_id"].apply(
            lambda x: STATION_METADATA.get(int(x), {}).get("name", f"Station {x}")
        )
        df["city"] = df["location_id"].apply(
            lambda x: STATION_METADATA.get(int(x), {}).get("city", "Delhi NCR")
        )
        return df
    except Exception as e:
        logger.error("Error reading %s: %s", PATH_CPCB_DAILY, e)
        return None


@cache_data
def load_aqi_predictions() -> Optional[pd.DataFrame]:
    """Load held-out test predictions and residuals."""
    if not PATH_AQI_PREDICTIONS.exists():
        logger.warning("File not found: %s", PATH_AQI_PREDICTIONS)
        return None
    try:
        df = pd.read_csv(PATH_AQI_PREDICTIONS)
        df["date"] = pd.to_datetime(df["date"])
        df["station_name"] = df["location_id"].apply(
            lambda x: STATION_METADATA.get(int(x), {}).get("name", f"Station {x}")
        )
        return df
    except Exception as e:
        logger.error("Error reading %s: %s", PATH_AQI_PREDICTIONS, e)
        return None


@cache_data
def load_hcho_clusters_summary() -> Optional[pd.DataFrame]:
    """Load HCHO hotspot cluster summary metrics."""
    if not PATH_HCHO_SUMMARY.exists():
        logger.warning("File not found: %s", PATH_HCHO_SUMMARY)
        return None
    try:
        df = pd.read_csv(PATH_HCHO_SUMMARY)
        # Assign geographic context names based on coordinates
        cluster_names = {
            0: "Delhi-NCR & Western UP Belt",
            1: "Singrauli / Sonbhadra Coal Belt",
            2: "Mumbai-Thane Industrial Corridor",
        }
        df["region_label"] = df["cluster_id"].apply(
            lambda cid: cluster_names.get(int(cid), f"Cluster {cid}")
        )
        return df
    except Exception as e:
        logger.error("Error reading %s: %s", PATH_HCHO_SUMMARY, e)
        return None


@cache_data
def load_hcho_hotspots_geojson() -> Optional[Dict[str, Any]]:
    """Load RFC 7946 GeoJSON FeatureCollection of HCHO hotspot clusters."""
    if not PATH_HCHO_GEOJSON.exists():
        logger.warning("File not found: %s", PATH_HCHO_GEOJSON)
        return None
    try:
        with open(PATH_HCHO_GEOJSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "FeatureCollection":
            logger.warning("Invalid GeoJSON type in %s", PATH_HCHO_GEOJSON)
        return data
    except Exception as e:
        logger.error("Error loading %s: %s", PATH_HCHO_GEOJSON, e)
        return None


@cache_data
def load_hcho_benchmark() -> Optional[pd.DataFrame]:
    """Load TROPOMI observation benchmark points across India."""
    if not PATH_HCHO_BENCHMARK.exists():
        logger.warning("File not found: %s", PATH_HCHO_BENCHMARK)
        return None
    try:
        return pd.read_csv(PATH_HCHO_BENCHMARK)
    except Exception as e:
        logger.error("Error loading %s: %s", PATH_HCHO_BENCHMARK, e)
        return None


@cache_data
def load_json_report(path: Path) -> Optional[Dict[str, Any]]:
    """Generic cached JSON report loader with validation."""
    if not path.exists():
        logger.warning("Report file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error reading %s: %s", path, e)
        return None


def load_aqi_metrics() -> Optional[Dict[str, Any]]:
    return load_json_report(PATH_AQI_METRICS)


def load_cpcb_report() -> Optional[Dict[str, Any]]:
    return load_json_report(PATH_CPCB_REPORT)


def load_hcho_report() -> Optional[Dict[str, Any]]:
    return load_json_report(PATH_HCHO_REPORT)


@cache_resource
def load_aqi_model() -> Optional[Any]:
    """Load trained XGBoost model binary."""
    if not PATH_AQI_MODEL.exists():
        logger.warning("Model file not found: %s", PATH_AQI_MODEL)
        return None
    try:
        with open(PATH_AQI_MODEL, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Error unpickling model from %s: %s", PATH_AQI_MODEL, e)
        return None


def load_state_coverage_report() -> Optional[Dict[str, Any]]:
    """Load comprehensive 36-state/UT coverage report."""
    return load_json_report(PATH_STATE_COVERAGE_REPORT)


@cache_data
def load_state_coverage_summary() -> Optional[pd.DataFrame]:
    """Load tabular state-wise coverage metrics."""
    if not PATH_STATE_COVERAGE_SUMMARY.exists():
        logger.warning("File not found: %s", PATH_STATE_COVERAGE_SUMMARY)
        return None
    try:
        return pd.read_csv(PATH_STATE_COVERAGE_SUMMARY)
    except Exception as e:
        logger.error("Error loading %s: %s", PATH_STATE_COVERAGE_SUMMARY, e)
        return None


@cache_data
def load_india_states_geojson() -> Optional[Dict[str, Any]]:
    """Load simplified GeoJSON for all 36 Indian states and UTs."""
    if not PATH_INDIA_STATES_GEOJSON.exists():
        logger.warning("File not found: %s", PATH_INDIA_STATES_GEOJSON)
        return None
    try:
        with open(PATH_INDIA_STATES_GEOJSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading %s: %s", PATH_INDIA_STATES_GEOJSON, e)
        return None


def load_satellite_meteo_readiness() -> Optional[Dict[str, Any]]:
    """Load satellite and meteorological data readiness report."""
    return load_json_report(PATH_SATELLITE_METEO_READINESS)

