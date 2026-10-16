"""
Surface AQI Estimation Model using XGBoost.

This module implements:
1. Feature engineering from station monitoring data (cyclical temporal encodings,
   spatial features, rolling and lag statistics).
2. Leakage-free chronological train/test splitting.
3. XGBoost regression model training and hyperparameter configuration.
4. Evaluation comparing XGBoost against a naive baseline (historical median / mean).
5. Comprehensive metrics (MAE, RMSE, R²) and artifact serialization.

Scientific & Methodological Notes
---------------------------------
- The model target is defined as ground-level CPCB AQI (or PM2.5 sub-index).
- Satellite column observations cannot be directly equated with surface concentrations
  without boundary layer height, temperature, and aerosol vertical profile scaling.
- In Phase 1, the model is trained on verified ground monitoring stations; national
  coverage is explicitly NOT claimed until satellite-meteorology co-location is integrated.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "aqi",
    include_lags: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Engineer temporal, spatial, and autoregressive features for AQI estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: date, latitude, longitude, and target_col.
    target_col : str
        Target column name (e.g. 'aqi' or 'pm25').
    include_lags : bool
        Whether to generate chronological lag features per station.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Transformed dataframe and list of feature column names.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location_id", "date"]).reset_index(drop=True)

    # 1. Cyclical calendar encodings (captures seasonal pollution trends in India)
    day_of_year = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Day of week (traffic patterns)
    day_of_week = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df["is_weekend"] = (day_of_week >= 5).astype(int)

    # 2. Spatial coordinates
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    feature_cols = [
        "latitude",
        "longitude",
        "doy_sin",
        "doy_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]

    # 3. Non-leaking chronological lag features (per station)
    if include_lags and target_col in df.columns:
        # Lag 1 day
        df["lag_1d"] = df.groupby("location_id")[target_col].shift(1)
        # 3-day rolling mean excluding the current day
        df["rolling_3d_mean"] = (
            df.groupby("location_id")[target_col]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        feature_cols.extend(["lag_1d", "rolling_3d_mean"])

    # Filter out rows where target is missing
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    return df, feature_cols


def chronological_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and held-out test sets.

    Ensures zero temporal leakage: training data strictly precedes test data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a 'date' column.
    test_ratio : float
        Fraction of unique dates allocated to the held-out test set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    if "date" not in df.columns:
        raise ValueError("Dataframe must contain 'date' column for chronological split.")

    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    if n_dates < 5:
        raise ValueError(f"Insufficient unique dates ({n_dates}) for chronological split.")

    split_idx = int(n_dates * (1.0 - test_ratio))
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    logger.info(
        "Chronological split at %s: Train=%d samples, Test=%d samples",
        split_date, len(train_df), len(test_df),
    )
    return train_df, test_df


def train_baseline_model(
    y_train: pd.Series,
) -> Dict[str, float]:
    """Train simple empirical baseline models (mean and median)."""
    return {
        "mean_baseline": float(y_train.mean()),
        "median_baseline": float(y_train.median()),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate standard regression metrics: MAE, RMSE, and R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    if len(y_t) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
    }


def train_aqi_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train and evaluate an XGBoost regression model for AQI estimation.

    Parameters
    ----------
    X_train, y_train : Features and target for training.
    X_test, y_test : Features and target for held-out evaluation.
    n_estimators : int
        Number of boosting trees.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Boosting shrinkage parameter.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[xgb.XGBRegressor, Dict[str, Any]]
        Fitted model and comprehensive evaluation metrics dict.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = evaluate_predictions(y_train.values, y_pred_train)
    test_metrics = evaluate_predictions(y_test.values, y_pred_test)

    # Baseline comparison (predicting median of training set)
    median_val = float(y_train.median())
    baseline_pred_test = np.full_like(y_test.values, fill_value=median_val, dtype=float)
    baseline_metrics = evaluate_predictions(y_test.values, baseline_pred_test)

    # Feature importance
    feature_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(X_train.columns, model.feature_importances_)
    }
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    results = {
        "model_type": "XGBoost Regressor",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_comparison": {
            "strategy": "Historical Median Baseline",
            "baseline_value": round(median_val, 2),
            "baseline_test_metrics": baseline_metrics,
            "mae_reduction_percent": round(
                ((baseline_metrics["mae"] - test_metrics["mae"]) / baseline_metrics["mae"]) * 100, 2
            ) if baseline_metrics["mae"] > 0 else 0.0,
        },
        "feature_importance": feature_importance,
    }

    return model, results


def save_pipeline_artifacts(
    model: xgb.XGBRegressor,
    metrics: Dict[str, Any],
    test_df: pd.DataFrame,
    y_pred_test: np.ndarray,
    target_col: str,
    output_dir: Path,
) -> Dict[str, str]:
    """Serialize model artifacts, predictions, and metrics."""
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "outputs" / "metrics"
    preds_dir = output_dir / "outputs" / "predictions"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model pickle & JSON
    model_pkl_path = models_dir / "aqi_xgboost.pkl"
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model, f)

    # 2. Metrics JSON
    metrics_path = metrics_dir / "aqi_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 3. Test predictions CSV
    preds_df = test_df[["location_id", "date", "latitude", "longitude", target_col]].copy()
    preds_df["predicted_aqi"] = np.round(y_pred_test, 1)
    preds_df["residual"] = np.round(preds_df[target_col] - preds_df["predicted_aqi"], 1)
    preds_path = preds_dir / "aqi_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    return {
        "model_artifact": str(model_pkl_path),
        "metrics_artifact": str(metrics_path),
        "predictions_artifact": str(preds_path),
    }
