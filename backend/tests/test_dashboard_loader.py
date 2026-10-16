"""Unit tests for the dashboard data loader layer and artifact integrity."""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_loader import (
    load_cpcb_daily_aqi,
    load_aqi_predictions,
    load_hcho_clusters_summary,
    load_hcho_hotspots_geojson,
    load_hcho_benchmark,
    load_aqi_metrics,
    load_cpcb_report,
    load_hcho_report,
    load_aqi_model,
)


def test_load_cpcb_daily_aqi():
    df = load_cpcb_daily_aqi()
    assert df is not None
    assert not df.empty
    assert "date" in df.columns
    assert "aqi" in df.columns
    assert "station_name" in df.columns
    assert len(df) >= 600


def test_load_aqi_predictions():
    df = load_aqi_predictions()
    assert df is not None
    assert not df.empty
    assert "predicted_aqi" in df.columns
    assert "residual" in df.columns
    assert "date" in df.columns
    assert len(df) >= 100


def test_load_hcho_clusters_summary():
    df = load_hcho_clusters_summary()
    assert df is not None
    assert not df.empty
    assert "cluster_id" in df.columns
    assert "centroid_lat" in df.columns
    assert "centroid_lon" in df.columns
    assert "region_label" in df.columns
    assert len(df) == 3


def test_load_hcho_hotspots_geojson():
    geojson = load_hcho_hotspots_geojson()
    assert geojson is not None
    assert geojson.get("type") == "FeatureCollection"
    features = geojson.get("features", [])
    assert len(features) == 3
    for f in features:
        assert f.get("type") == "Feature"
        assert f.get("geometry", {}).get("type") == "Point"
        coords = f.get("geometry", {}).get("coordinates", [])
        assert len(coords) == 2
        # RFC 7946: [lon, lat]
        lon, lat = coords
        assert 68.0 <= lon <= 97.5
        assert 6.5 <= lat <= 37.5


def test_load_reports():
    cpcb = load_cpcb_report()
    assert cpcb is not None
    assert "raw_data" in cpcb
    assert "aqi_summary" in cpcb

    hcho = load_hcho_report()
    assert hcho is not None
    assert "clustering_parameters" in hcho
    assert hcho.get("num_clusters") == 3

    metrics = load_aqi_metrics()
    assert metrics is not None
    assert "train_metrics" in metrics
    assert "test_metrics" in metrics
    assert "baseline_comparison" in metrics


def test_load_aqi_model():
    model = load_aqi_model()
    assert model is not None
    assert hasattr(model, "predict")
