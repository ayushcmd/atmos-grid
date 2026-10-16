"""Tests for Surface AQI Estimation model, feature engineering, and metrics."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.models.aqi_model import (
    chronological_split,
    engineer_features,
    evaluate_predictions,
    train_baseline_model,
    train_aqi_xgboost,
)


@pytest.fixture
def sample_aqi_data() -> pd.DataFrame:
    """Create synthetic multi-day, multi-station AQI data."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    records = []
    for loc_id in [101, 102]:
        for d in dates:
            base_aqi = 120.0 + 30.0 * np.sin(d.dayofyear / 20.0)
            noise = np.random.normal(0, 10.0)
            records.append({
                "location_id": loc_id,
                "date": d.strftime("%Y-%m-%d"),
                "latitude": 28.6 if loc_id == 101 else 19.1,
                "longitude": 77.2 if loc_id == 101 else 72.8,
                "pm25": max(5.0, (base_aqi + noise) * 0.4),
                "aqi": max(10.0, base_aqi + noise),
            })
    return pd.DataFrame(records)


def test_engineer_features(sample_aqi_data):
    df_feat, feature_cols = engineer_features(sample_aqi_data, target_col="aqi", include_lags=True)
    assert "doy_sin" in feature_cols
    assert "doy_cos" in feature_cols
    assert "month_sin" in feature_cols
    assert "latitude" in feature_cols
    assert "lag_1d" in feature_cols
    assert "rolling_3d_mean" in feature_cols
    assert not df_feat.empty


def test_chronological_split(sample_aqi_data):
    train_df, test_df = chronological_split(sample_aqi_data, test_ratio=0.20)
    assert len(train_df) > 0
    assert len(test_df) > 0
    # Strict temporal boundary check: max(train) < min(test)
    assert pd.to_datetime(train_df["date"]).max() < pd.to_datetime(test_df["date"]).min()


def test_evaluate_predictions():
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([105.0, 145.0, 195.0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["mae"] == 5.0
    assert metrics["rmse"] == 5.0
    assert metrics["r2"] > 0.95


def test_train_aqi_xgboost(sample_aqi_data):
    df_feat, feature_cols = engineer_features(sample_aqi_data, target_col="aqi", include_lags=True)
    df_clean = df_feat.dropna().reset_index(drop=True)
    train_df, test_df = chronological_split(df_clean, test_ratio=0.25)

    X_train = train_df[feature_cols]
    y_train = train_df["aqi"]
    X_test = test_df[feature_cols]
    y_test = test_df["aqi"]

    model, results = train_aqi_xgboost(
        X_train, y_train, X_test, y_test,
        n_estimators=30, max_depth=3, learning_rate=0.1
    )

    assert results["model_type"] == "XGBoost Regressor"
    assert "mae" in results["test_metrics"]
    assert "rmse" in results["test_metrics"]
    assert "r2" in results["test_metrics"]
    assert "baseline_comparison" in results
    assert len(results["feature_importance"]) == len(feature_cols)
