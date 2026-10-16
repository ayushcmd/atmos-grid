"""Tests for CPCB data quality, AQI calculation, and daily aggregation."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.data.fetch_cpcb_openaq import (
    CPCB_AQI_BREAKPOINTS,
    aggregate_daily_pollutants,
    calculate_aqi,
    calculate_sub_index,
    compute_daily_aqi,
)


class TestSubIndexCalculation:
    """Test CPCB sub-index calculations against known breakpoint values."""

    def test_pm25_good_range(self):
        """PM2.5 = 15 µg/m³ should give sub-index in Good (0-50) range."""
        si = calculate_sub_index(15, "pm25")
        assert si is not None
        assert 0 <= si <= 50

    def test_pm25_boundary(self):
        """PM2.5 = 30 should give exactly 50 (boundary of Good range)."""
        si = calculate_sub_index(30, "pm25")
        assert si == 50.0

    def test_pm25_poor_range(self):
        """PM2.5 = 75 µg/m³ should give sub-index in Poor (101-200) range."""
        si = calculate_sub_index(75, "pm25")
        assert si is not None
        assert 101 <= si <= 200

    def test_pm25_severe_range(self):
        """PM2.5 = 200 µg/m³ should give sub-index in Severe (301-400) range."""
        si = calculate_sub_index(200, "pm25")
        assert si is not None
        assert 301 <= si <= 400

    def test_pm10_moderate(self):
        """PM10 = 80 µg/m³ should be in Satisfactory (51-100) range."""
        si = calculate_sub_index(80, "pm10")
        assert si is not None
        assert 51 <= si <= 100

    def test_no2_good(self):
        """NO2 = 20 µg/m³ should be in Good range."""
        si = calculate_sub_index(20, "no2")
        assert si is not None
        assert 0 <= si <= 50

    def test_negative_concentration(self):
        """Negative concentrations should return None."""
        assert calculate_sub_index(-5, "pm25") is None

    def test_nan_concentration(self):
        """NaN should return None."""
        assert calculate_sub_index(float("nan"), "pm25") is None

    def test_unknown_pollutant(self):
        """Unknown pollutant should return None."""
        assert calculate_sub_index(50, "unknown_gas") is None

    def test_exceeds_max_breakpoint(self):
        """Concentration above max breakpoint should return None."""
        # PM2.5 max breakpoint is 500
        assert calculate_sub_index(600, "pm25") is None

    def test_zero_concentration(self):
        """Zero concentration should give sub-index 0."""
        si = calculate_sub_index(0, "pm25")
        assert si == 0.0

    def test_co_units(self):
        """CO breakpoints use mg/m³. CO = 1.5 should be Satisfactory."""
        si = calculate_sub_index(1.5, "co")
        assert si is not None
        assert 51 <= si <= 100


class TestOverallAQI:
    """Test overall AQI calculation (max of sub-indices)."""

    def test_single_pollutant(self):
        """AQI with only PM2.5 should equal PM2.5 sub-index."""
        result = calculate_aqi({"pm25": 45})
        assert result["aqi"] is not None
        assert result["dominant_pollutant"] == "pm25"
        assert result["n_valid"] == 1

    def test_multiple_pollutants_pm25_dominant(self):
        """PM2.5 = 100 should dominate over NO2 = 20."""
        result = calculate_aqi({"pm25": 100, "no2": 20})
        assert result["dominant_pollutant"] == "pm25"
        assert result["n_valid"] == 2
        # PM2.5=100 is in Poor range (101-200), NO2=20 is Good (0-50)
        assert result["aqi"] > result["sub_indices"]["no2"]

    def test_empty_input(self):
        """Empty concentrations should return None AQI."""
        result = calculate_aqi({})
        assert result["aqi"] is None
        assert result["n_valid"] == 0

    def test_all_nan_input(self):
        """All-NaN concentrations should return None AQI."""
        result = calculate_aqi({"pm25": float("nan"), "pm10": float("nan")})
        assert result["aqi"] is None
        assert result["n_valid"] == 0

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid should compute AQI from valid ones."""
        result = calculate_aqi({"pm25": 50, "no2": float("nan"), "so2": -10})
        assert result["aqi"] is not None
        assert result["n_valid"] == 1
        assert result["dominant_pollutant"] == "pm25"


class TestDailyAggregation:
    """Test daily aggregation of hourly measurements."""

    def _make_hourly_data(self, n_hours=24, base_value=50.0) -> pd.DataFrame:
        """Create synthetic hourly data for testing."""
        dates = pd.date_range("2024-01-01", periods=n_hours, freq="h")
        return pd.DataFrame({
            "location_id": [1] * n_hours,
            "datetime": dates,
            "latitude": [28.7] * n_hours,
            "longitude": [77.1] * n_hours,
            "parameter": ["pm25"] * n_hours,
            "value": np.random.normal(base_value, 10, n_hours).clip(0),
            "unit": ["µg/m³"] * n_hours,
        })

    def test_full_day_aggregation(self):
        """24 hours of data should produce one daily record."""
        df = self._make_hourly_data(24, 50.0)
        daily = aggregate_daily_pollutants(df)
        assert len(daily) == 1
        assert "pm25" in daily.columns

    def test_insufficient_hours_excluded(self):
        """Fewer than 16 hours should be excluded."""
        df = self._make_hourly_data(10, 50.0)
        daily = aggregate_daily_pollutants(df)
        assert len(daily) == 0

    def test_sixteen_hours_included(self):
        """Exactly 16 hours should be included."""
        df = self._make_hourly_data(16, 50.0)
        daily = aggregate_daily_pollutants(df)
        assert len(daily) == 1

    def test_empty_input(self):
        """Empty DataFrame should return empty."""
        daily = aggregate_daily_pollutants(pd.DataFrame())
        assert daily.empty

    def test_multiple_pollutants_pivoted(self):
        """Multiple pollutants should be pivoted to columns."""
        pm25 = self._make_hourly_data(24, 50.0)
        pm10 = self._make_hourly_data(24, 80.0)
        pm10["parameter"] = "pm10"
        combined = pd.concat([pm25, pm10], ignore_index=True)
        daily = aggregate_daily_pollutants(combined)
        assert len(daily) == 1
        assert "pm25" in daily.columns
        assert "pm10" in daily.columns


class TestComputeDailyAQI:
    """Test AQI computation on daily aggregated data."""

    def test_aqi_computed(self):
        """Daily data with PM2.5 should produce valid AQI."""
        daily = pd.DataFrame({
            "location_id": [1],
            "date": [pd.Timestamp("2024-01-01").date()],
            "latitude": [28.7],
            "longitude": [77.1],
            "pm25": [75.0],
            "pm10": [120.0],
        })
        result = compute_daily_aqi(daily)
        assert "aqi" in result.columns
        assert result["aqi"].iloc[0] is not None
        assert result["aqi"].iloc[0] > 0

    def test_empty_daily(self):
        """Empty input should return empty."""
        result = compute_daily_aqi(pd.DataFrame())
        assert result.empty
