"""Unit and integration tests for HCHO hotspot clustering and pipeline.

Tests cover:
1. Haversine distance calculation and mathematical properties
2. Data-driven threshold selection logic
3. DBSCAN spatial clustering with synthetic clusters of known coordinates
4. RFC 7946 GeoJSON export validity
5. Edge cases: empty data, single point, missing columns, all-NaN data
6. Scientific unit conversions
7. Graceful Earth Engine initialization and access handling
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from backend.src.data.fetch_tropomi_hcho import (
    MOL_M2_TO_MOLECULES_CM2,
    build_quality_report,
    test_gee_access,
)
from backend.src.models.hcho_hotspots import (
    EARTH_RADIUS_KM,
    compute_concentration_distribution,
    detect_hcho_hotspots,
    export_hotspots_geojson,
    export_results,
    haversine_distance,
    validate_input_data,
)


# =====================================================================
# Fixtures and Helpers
# =====================================================================


def generate_synthetic_points(
    center_lat: float,
    center_lon: float,
    num_points: int,
    radius_km: float,
    hcho_mean: float,
    hcho_std: float,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic spatial points around a center with specified radius and HCHO values."""
    rng = np.random.default_rng(random_seed)

    # 1 degree latitude ~ 111 km
    # 1 degree longitude ~ 111 km * cos(lat)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))

    # Random angles and distances
    angles = rng.uniform(0, 2 * math.pi, num_points)
    radii = rng.uniform(0, radius_km, num_points)

    d_lats = (radii * np.sin(angles)) / km_per_deg_lat
    d_lons = (radii * np.cos(angles)) / km_per_deg_lon

    lats = center_lat + d_lats
    lons = center_lon + d_lons
    hcho_vals = rng.normal(hcho_mean, hcho_std, num_points)
    hcho_vals = np.clip(hcho_vals, 1e14, None)

    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "hcho_molecules_cm2": hcho_vals,
        "hcho_mol_m2": hcho_vals / MOL_M2_TO_MOLECULES_CM2,
    })


@pytest.fixture
def synthetic_cluster_data() -> Tuple[pd.DataFrame, float, float, float, float]:
    """Create synthetic dataset with 2 known distinct clusters plus background noise.

    Cluster 1: Near Delhi (28.61, 77.23) - 15 points, elevated HCHO
    Cluster 2: Near Mumbai (19.07, 72.87) - 15 points, elevated HCHO
    Background: 100 scattered points across India with lower HCHO values
    """
    c1_lat, c1_lon = 28.61, 77.23
    c2_lat, c2_lon = 19.07, 72.87

    # Elevated clusters: mean 2.5e16 molec/cm²
    df_c1 = generate_synthetic_points(
        center_lat=c1_lat,
        center_lon=c1_lon,
        num_points=15,
        radius_km=15.0,
        hcho_mean=2.5e16,
        hcho_std=1.0e15,
        random_seed=101,
    )
    df_c2 = generate_synthetic_points(
        center_lat=c2_lat,
        center_lon=c2_lon,
        num_points=15,
        radius_km=15.0,
        hcho_mean=2.2e16,
        hcho_std=1.0e15,
        random_seed=102,
    )

    # Background points: lower values (mean 5.0e15 molec/cm²)
    rng = np.random.default_rng(202)
    bg_lats = rng.uniform(10.0, 30.0, 100)
    bg_lons = rng.uniform(70.0, 88.0, 100)
    bg_hcho = rng.normal(5.0e15, 8.0e14, 100)
    bg_hcho = np.clip(bg_hcho, 1e14, None)

    df_bg = pd.DataFrame({
        "latitude": bg_lats,
        "longitude": bg_lons,
        "hcho_molecules_cm2": bg_hcho,
        "hcho_mol_m2": bg_hcho / MOL_M2_TO_MOLECULES_CM2,
    })

    combined_df = pd.concat([df_c1, df_c2, df_bg], ignore_index=True)
    return combined_df, c1_lat, c1_lon, c2_lat, c2_lon


# =====================================================================
# 1. Haversine Distance Tests
# =====================================================================


class TestHaversineDistance:
    """Test great-circle distance calculations."""

    def test_same_point_zero_distance(self):
        """Distance between a point and itself must be zero."""
        dist = haversine_distance(28.6139, 77.2090, 28.6139, 77.2090)
        assert math.isclose(dist, 0.0, abs_tol=1e-6)

    def test_known_equator_to_pole_distance(self):
        """Quarter meridian: (0, 0) to (90, 0) should be π/2 * R ≈ 10007.54 km."""
        expected = (math.pi / 2.0) * EARTH_RADIUS_KM
        dist = haversine_distance(0.0, 0.0, 90.0, 0.0)
        assert math.isclose(dist, expected, rel_tol=1e-4)

    def test_delhi_to_mumbai_distance(self):
        """Known geographical distance between Delhi and Mumbai is ~1148-1155 km."""
        delhi_lat, delhi_lon = 28.6139, 77.2090
        mumbai_lat, mumbai_lon = 19.0760, 72.8777
        dist = haversine_distance(delhi_lat, delhi_lon, mumbai_lat, mumbai_lon)
        assert 1140.0 <= dist <= 1160.0

    def test_distance_symmetry(self):
        """Distance from A to B must equal distance from B to A."""
        d1 = haversine_distance(12.9716, 77.5946, 22.5726, 88.3639)
        d2 = haversine_distance(22.5726, 88.3639, 12.9716, 77.5946)
        assert math.isclose(d1, d2, abs_tol=1e-6)


# =====================================================================
# 2. Distribution and Threshold Selection Tests
# =====================================================================


class TestDistributionAndThreshold:
    """Test statistical distribution and threshold selection."""

    def test_distribution_statistics(self):
        """Test calculation of min, max, mean, median, P75, P90, P95, P99."""
        data = pd.Series(np.linspace(10.0, 100.0, 91))  # 10, 11, ..., 100
        stats = compute_concentration_distribution(data)

        assert math.isclose(stats["min"], 10.0)
        assert math.isclose(stats["max"], 100.0)
        assert math.isclose(stats["mean"], 55.0)
        assert math.isclose(stats["median"], 55.0)
        assert math.isclose(stats["p75"], 77.5)
        assert math.isclose(stats["p90"], 91.0)
        assert math.isclose(stats["p95"], 95.5)
        assert math.isclose(stats["p99"], 99.1)

    def test_threshold_selection_logic(self):
        """Test that data-driven P90 threshold isolates top 10% values."""
        rng = np.random.default_rng(42)
        vals = rng.normal(1e16, 2e15, 1000)
        df = pd.DataFrame({
            "latitude": rng.uniform(10, 30, 1000),
            "longitude": rng.uniform(70, 90, 1000),
            "hcho_molecules_cm2": vals,
        })

        expected_p90 = np.percentile(vals, 90.0)
        clusters_df, elevated_df, report = detect_hcho_hotspots(
            df,
            threshold_percentile=90.0,
            eps_km=50.0,
            min_samples=5,
        )

        assert math.isclose(
            report["clustering_parameters"]["threshold_value_molecules_cm2"],
            expected_p90,
            rel_tol=1e-5,
        )
        assert len(elevated_df) == 100  # Exactly top 10% of 1000 points
        assert (elevated_df["hcho_molecules_cm2"] >= expected_p90).all()


# =====================================================================
# 3. DBSCAN Clustering with Synthetic Data
# =====================================================================


class TestHotspotClustering:
    """Test DBSCAN spatial clustering on known synthetic clusters."""

    def test_synthetic_clusters_detection(self, synthetic_cluster_data):
        """Verify that DBSCAN detects exactly the 2 known clusters near Delhi and Mumbai."""
        df, c1_lat, c1_lon, c2_lat, c2_lon = synthetic_cluster_data

        clusters_df, elevated_df, report = detect_hcho_hotspots(
            data=df,
            eps_km=50.0,
            min_samples=5,
            threshold_percentile=75.0,  # Elevated clusters represent top ~23% of points
        )

        # Must detect exactly 2 clusters
        assert len(clusters_df) == 2, f"Expected 2 clusters, got {len(clusters_df)}"
        assert report["num_clusters"] == 2

        # Check centroids are close to original points (within 25 km)
        found_c1 = False
        found_c2 = False

        for _, row in clusters_df.iterrows():
            d_c1 = haversine_distance(row["centroid_lat"], row["centroid_lon"], c1_lat, c1_lon)
            d_c2 = haversine_distance(row["centroid_lat"], row["centroid_lon"], c2_lat, c2_lon)

            if d_c1 < 25.0:
                found_c1 = True
                assert row["pixel_count"] >= 5
                assert row["mean_hcho_molecules_cm2"] > 2.0e16
            elif d_c2 < 25.0:
                found_c2 = True
                assert row["pixel_count"] >= 5
                assert row["mean_hcho_molecules_cm2"] > 2.0e16

        assert found_c1, "Cluster near Delhi was not detected within tolerance"
        assert found_c2, "Cluster near Mumbai was not detected within tolerance"


# =====================================================================
# 4. GeoJSON Validity and RFC 7946 Compliance
# =====================================================================


class TestGeoJSONExport:
    """Test GeoJSON validity and RFC 7946 specifications."""

    def test_geojson_validity(self, tmp_path, synthetic_cluster_data):
        """Ensure exported GeoJSON is valid JSON and adheres strictly to RFC 7946."""
        df, _, _, _, _ = synthetic_cluster_data
        clusters_df, _, _ = detect_hcho_hotspots(df, threshold_percentile=75.0)

        out_geojson = tmp_path / "test_hotspots.geojson"
        geojson_dict = export_hotspots_geojson(clusters_df, out_geojson)

        # 1. File exists and parses as valid JSON
        assert out_geojson.exists()
        with open(out_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. RFC 7946 root structure
        assert data.get("type") == "FeatureCollection"
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) == len(clusters_df)

        # 3. Feature validation
        for feat in data["features"]:
            assert feat.get("type") == "Feature"

            # Geometry checks
            geom = feat.get("geometry", {})
            assert geom.get("type") == "Point"
            coords = geom.get("coordinates")
            assert isinstance(coords, list) and len(coords) == 2

            lon, lat = coords
            # Longitude comes first in RFC 7946 Point coordinates
            assert -180.0 <= lon <= 180.0
            assert -90.0 <= lat <= 90.0

            # Properties checks
            props = feat.get("properties", {})
            assert "cluster_id" in props
            assert "pixel_count" in props
            assert "mean_hcho_molecules_cm2" in props
            assert "max_hcho_molecules_cm2" in props
            assert props["pixel_count"] > 0

    def test_export_all_results(self, tmp_path):
        """Test export_results creates GeoJSON, summary CSV, and metrics JSON."""
        clusters_df = pd.DataFrame([{
            "cluster_id": 0,
            "centroid_lat": 28.61,
            "centroid_lon": 77.23,
            "pixel_count": 10,
            "mean_hcho_molecules_cm2": 2.5e16,
            "max_hcho_molecules_cm2": 2.8e16,
            "min_hcho_molecules_cm2": 2.2e16,
        }])
        report = {
            "num_clusters": 1,
            "total_pixels": 100,
            "scientific_disclaimer": "Test disclaimer",
        }

        geo_path = tmp_path / "out.geojson"
        csv_path = tmp_path / "out.csv"
        json_path = tmp_path / "out.json"

        export_results(
            clusters_df=clusters_df,
            report=report,
            geojson_path=geo_path,
            summary_path=csv_path,
            metrics_path=json_path,
        )

        assert geo_path.exists()
        assert csv_path.exists()
        assert json_path.exists()

        # Check CSV content
        read_csv = pd.read_csv(csv_path)
        assert len(read_csv) == 1
        assert read_csv.iloc[0]["cluster_id"] == 0


# =====================================================================
# 5. Edge Cases
# =====================================================================


class TestEdgeCases:
    """Test boundary and error edge cases."""

    def test_empty_dataframe(self, tmp_path):
        """Empty input DataFrame should produce 0 clusters and valid empty GeoJSON."""
        empty_df = pd.DataFrame()
        clusters_df, elevated_df, report = detect_hcho_hotspots(empty_df)

        assert len(clusters_df) == 0
        assert len(elevated_df) == 0
        assert report["num_clusters"] == 0

        # Export must not fail
        geo_path = tmp_path / "empty.geojson"
        export_hotspots_geojson(clusters_df, geo_path)
        assert geo_path.exists()
        with open(geo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 0

    def test_single_point(self):
        """A single point cannot form a cluster with min_samples=5."""
        single_df = pd.DataFrame({
            "latitude": [28.61],
            "longitude": [77.23],
            "hcho_molecules_cm2": [2.5e16],
        })
        clusters_df, elevated_df, report = detect_hcho_hotspots(
            single_df, min_samples=5
        )

        assert len(clusters_df) == 0
        assert report["num_clusters"] == 0
        assert report["noise_pixels"] == 1

    def test_missing_coordinates(self):
        """Missing required columns must raise ValueError."""
        invalid_df = pd.DataFrame({
            "lat": [28.61],  # Wrong column name
            "longitude": [77.23],
            "hcho_molecules_cm2": [2.5e16],
        })
        with pytest.raises(ValueError, match="Input data must contain 'latitude' and 'longitude'"):
            validate_input_data(invalid_df)

    def test_all_nan_data(self):
        """All-NaN concentration column should raise ValueError."""
        nan_df = pd.DataFrame({
            "latitude": [28.61, 28.62],
            "longitude": [77.23, 77.24],
            "hcho_molecules_cm2": [np.nan, np.nan],
        })
        with pytest.raises(ValueError, match="All values in concentration column .* are NaN"):
            validate_input_data(nan_df)

    def test_mol_m2_conversion(self):
        """When given hcho_mol_m2, it should automatically convert to molecules/cm²."""
        raw_val = 1.0e-4  # mol/m²
        expected_molec = raw_val * MOL_M2_TO_MOLECULES_CM2
        df = pd.DataFrame({
            "latitude": [28.61],
            "longitude": [77.23],
            "hcho_mol_m2": [raw_val],
        })
        clusters_df, elevated_df, report = detect_hcho_hotspots(df, min_samples=5)

        assert "hcho_molecules_cm2" in elevated_df.columns
        assert math.isclose(
            elevated_df.iloc[0]["hcho_molecules_cm2"], expected_molec, rel_tol=1e-5
        )

    def test_uniform_data(self):
        """When all points have identical values, thresholding works cleanly."""
        df = pd.DataFrame({
            "latitude": np.linspace(20, 21, 10),
            "longitude": np.linspace(75, 76, 10),
            "hcho_molecules_cm2": [1.0e16] * 10,
        })
        clusters_df, elevated_df, report = detect_hcho_hotspots(
            df, eps_km=50.0, min_samples=3
        )
        assert len(elevated_df) == 10


# =====================================================================
# 6. Quality Report & GEE Access Graceful Handling
# =====================================================================


class TestQualityReportAndGEE:
    """Test quality report builder and GEE graceful degradation."""

    def test_build_quality_report(self):
        """Verify data quality report structure."""
        df = pd.DataFrame({
            "latitude": [20.0, 21.0],
            "longitude": [75.0, 76.0],
            "hcho_mol_m2": [1.0e-4, 1.5e-4],
            "hcho_molecules_cm2": [
                1.0e-4 * MOL_M2_TO_MOLECULES_CM2,
                1.5e-4 * MOL_M2_TO_MOLECULES_CM2,
            ],
        })
        report = build_quality_report(
            df,
            start_date="2024-05-01",
            end_date="2024-05-31",
            bbox=[68.0, 6.5, 97.5, 37.5],
        )

        assert report["row_count"] == 2
        assert report["valid_row_count"] == 2
        assert report["null_counts"]["hcho_mol_m2"] == 0
        assert report["date_range"]["start_date"] == "2024-05-01"
        assert "percentile_statistics_molecules_cm2" in report

    def test_gee_access_graceful_handling(self):
        """When GEE credentials are not configured or invalid, test_gee_access returns False without raising."""
        # Using a dummy project that won't authenticate
        result = test_gee_access(project="dummy-unauthenticated-project-999")
        assert result is False
