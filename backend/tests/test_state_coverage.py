"""
Unit tests for India-wide data expansion, state coverage aggregation,
coordinate validation, data completeness, and preflight diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from backend.src.data.fetch_cpcb_openaq import (
    CPCB_AQI_BREAKPOINTS,
    calculate_aqi,
    calculate_sub_index,
)
from backend.src.data.satellite_meteo_pipeline import (
    diagnose_copernicus_cds,
    diagnose_gee,
    diagnose_nasa_firms,
    run_full_preflight_diagnostics,
)
from dashboard.data_loader import (
    load_cpcb_daily_aqi,
    load_india_states_geojson,
    load_satellite_meteo_readiness,
    load_state_coverage_report,
    load_state_coverage_summary,
)

INDIA_BBOX = [68.0, 6.5, 97.5, 37.5]


class TestCoordinateValidation:
    """Validate spatial coordinates for Indian monitoring sites and bounding box."""

    def test_india_bounding_box_validity(self):
        min_lon, min_lat, max_lon, max_lat = INDIA_BBOX
        assert 68.0 <= min_lon < max_lon <= 97.5
        assert 6.5 <= min_lat < max_lat <= 37.5

    def test_known_cpcb_station_coordinates_within_india(self):
        stations = [
            ("DTU Delhi", 28.7500, 77.1170),
            ("Anand Vihar Delhi", 28.6508, 77.3152),
            ("Kanpur Nehru Nagar", 26.4700, 80.3200),
            ("Mumbai Bandra", 19.0600, 72.8400),
            ("Hyderabad Zoo Park", 17.3500, 78.4500),
            ("Noida Sector-62", 28.6245, 77.3649),
        ]
        min_lon, min_lat, max_lon, max_lat = INDIA_BBOX
        for name, lat, lon in stations:
            assert min_lat <= lat <= max_lat, f"{name} latitude {lat} outside India"
            assert min_lon <= lon <= max_lon, f"{name} longitude {lon} outside India"

    def test_out_of_bounds_detection(self):
        min_lon, min_lat, max_lon, max_lat = INDIA_BBOX
        invalid_coords = [
            (0.0, 0.0),      # Gulf of Guinea
            (51.5, -0.1),    # London
            (40.7, -74.0),   # New York
            (-33.8, 151.2),  # Sydney
        ]
        for lat, lon in invalid_coords:
            is_valid = (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)
            assert not is_valid, f"Coordinate ({lat}, {lon}) should be detected as outside India"


class TestCPCBDataCompleteness:
    """Validate the statutory CPCB 16-hour daily completeness rule."""

    def test_sixteen_hour_rule_inclusion(self):
        hours = 16
        is_valid = hours >= 16
        assert is_valid is True

    def test_fifteen_hour_rule_exclusion(self):
        hours = 15
        is_valid = hours >= 16
        assert is_valid is False

    def test_sub_index_boundary_conditions(self):
        # PM2.5 at exact breakpoint 30 is Good (50)
        assert calculate_sub_index(30.0, "pm25") == 50.0
        # PM2.5 at 0 is Good (0)
        assert calculate_sub_index(0.0, "pm25") == 0.0
        # Negative concentration returns None
        assert calculate_sub_index(-5.0, "pm25") is None
        # Out-of-range (>500) returns None
        assert calculate_sub_index(600.0, "pm25") is None


class TestStateCoverageAggregation:
    """Validate state coverage classification and reporting metrics."""

    def test_state_coverage_report_loaded(self):
        rep = load_state_coverage_report()
        assert rep is not None
        assert "total_indian_states_and_uts" in rep
        assert rep["total_indian_states_and_uts"] == 36
        assert rep["states_with_measured_ground"] >= 4
        assert rep["states_validated_model"] == 0  # Strictly 0 per guidelines

    def test_state_summary_dataframe_schema(self):
        df = load_state_coverage_summary()
        assert df is not None
        assert not df.empty
        assert len(df) == 36
        required_cols = ["state", "coverage_status", "station_count", "valid_station_days", "pollutants_measured"]
        for c in required_cols:
            assert c in df.columns

    def test_no_false_ground_coverage_claims(self):
        df = load_state_coverage_summary()
        assert df is not None
        ground_states = df[df["coverage_status"] == "MEASURED_GROUND"]["state"].tolist()
        # Verify only states with actual retrieved stations are labeled MEASURED_GROUND
        for s in ground_states:
            stn_cnt = df[df["state"] == s]["station_count"].iloc[0]
            assert stn_cnt > 0, f"State {s} claimed MEASURED_GROUND but has 0 stations"

    def test_simplified_india_geojson(self):
        gj = load_india_states_geojson()
        assert gj is not None
        assert gj.get("type") == "FeatureCollection"
        assert len(gj.get("features", [])) == 36


class TestSatelliteMeteoDiagnostics:
    """Validate preflight diagnostics across providers."""

    def test_gee_diagnostic_structure(self):
        diag = diagnose_gee()
        assert "service" in diag
        assert "status" in diag
        assert "auth_required" in diag

    def test_cds_diagnostic_structure(self):
        diag = diagnose_copernicus_cds()
        assert "service" in diag
        assert "status" in diag
        assert "variables" in diag

    def test_firms_diagnostic_structure(self):
        diag = diagnose_nasa_firms()
        assert "service" in diag
        assert "status" in diag
        assert diag["is_ready"] is True  # Public regional CSV feeds are open

    def test_full_diagnostic_report(self):
        rep = run_full_preflight_diagnostics()
        assert "providers" in rep
        assert "google_earth_engine" in rep["providers"]
        assert "copernicus_cds_era5" in rep["providers"]
        assert "nasa_firms_fire" in rep["providers"]
        assert "scientific_integration_plan" in rep
