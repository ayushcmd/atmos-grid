"""HCHO Hotspot Detection and Spatial Clustering Module.

This module analyzes Sentinel-5P TROPOMI tropospheric formaldehyde (HCHO)
column densities, identifies elevated concentration pixels using data-driven
statistical thresholds, and clusters them using Haversine-metric DBSCAN.

Scientific Assumptions and Caveats:
-----------------------------------
1. Concentration Patterns vs Emission Sources:
   Detected clusters represent spatial concentration patterns of elevated tropospheric
   formaldehyde (HCHO) column density, NOT confirmed emission sources.
   HCHO is an intermediate oxidation product of volatile organic compounds (VOCs)
   from both biogenic (e.g. isoprene) and anthropogenic (industrial, vehicle,
   biomass burning) sources. Column densities are influenced by non-linear photochemistry,
   boundary layer ventilation, OH radical availability, and horizontal atmospheric transport.
2. Coordinate and Metric Geometry:
   Coordinates are in WGS 84 (EPSG:4326). Clustering utilizes Great Circle (Haversine)
   distance on a spherical Earth with mean radius R = 6,371.0 km.
3. GeoJSON Specification:
   Exports conform strictly to RFC 7946, utilizing [longitude, latitude] coordinate ordering.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Logger configuration
logger = logging.getLogger(__name__)

# Constants
EARTH_RADIUS_KM: float = 6371.0
DEFAULT_EPS_KM: float = 50.0
DEFAULT_MIN_SAMPLES: int = 5
DEFAULT_THRESHOLD_PERCENTILE: float = 90.0
MOL_M2_TO_MOLECULES_CM2: float = 6.02214e19

# Scientific disclaimer
SCIENTIFIC_DISCLAIMER: str = (
    "Detected clusters represent spatial concentration patterns of elevated "
    "tropospheric formaldehyde (HCHO) column density, NOT confirmed direct emission sources. "
    "Formaldehyde is primarily a secondary photochemical oxidation product of VOC precursors. "
    "Atmospheric transport, meteorological conditions, and chemical lifetime affect column observations."
)


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """Calculate the great-circle distance between two points on Earth in kilometers.

    Uses the analytical Haversine formula:
        a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.
        radius_km: Earth radius in kilometers (default 6371.0 km).

    Returns:
        float: Great-circle distance in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    # Clip to avoid floating point domain errors in sqrt
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


def compute_concentration_distribution(series: pd.Series) -> Dict[str, float]:
    """Calculate statistical distribution metrics for HCHO concentration values.

    Metrics include: min, max, mean, median, P75, P90, P95, P99.

    Args:
        series: Pandas Series of concentration values.

    Returns:
        Dict[str, float]: Dictionary of statistical metrics.
    """
    clean = series.dropna()
    if clean.empty:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    arr = clean.to_numpy()
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def validate_input_data(df: pd.DataFrame) -> Tuple[str, str]:
    """Validate that the input DataFrame has necessary coordinate and concentration columns.

    Args:
        df: Pandas DataFrame to validate.

    Returns:
        Tuple[str, str]: Coordinate and primary concentration column names.

    Raises:
        ValueError: If required columns are missing or if DataFrame is invalid.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Input data must contain 'latitude' and 'longitude' columns.")

    if "hcho_molecules_cm2" in df.columns:
        concentration_col = "hcho_molecules_cm2"
    elif "hcho_mol_m2" in df.columns:
        concentration_col = "hcho_mol_m2"
    else:
        raise ValueError(
            "Input data must contain either 'hcho_molecules_cm2' or 'hcho_mol_m2' column."
        )

    # Check non-empty valid numbers
    valid_count = df[concentration_col].dropna().count()
    if len(df) > 0 and valid_count == 0:
        raise ValueError(f"All values in concentration column '{concentration_col}' are NaN.")

    return concentration_col, "latitude"


def detect_hcho_hotspots(
    data: Union[pd.DataFrame, str, Path],
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    observation_period: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Detect spatial clusters of elevated HCHO concentrations using DBSCAN with Haversine metric.

    Workflow:
    1. Loads input data and computes concentration distribution (min, max, mean, median, P75, P90, P95, P99).
    2. Determines data-driven threshold at threshold_percentile (default P90).
    3. Identifies elevated pixels exceeding threshold.
    4. Converts latitudes/longitudes to radians.
    5. Runs DBSCAN with metric='haversine' and eps = eps_km / 6371.0.
    6. Aggregates cluster statistics: centroid lat/lon, pixel count, mean/max concentrations.
    7. Generates comprehensive quality and metadata report.

    Args:
        data: Input DataFrame or file path to CSV containing TROPOMI HCHO data.
        eps_km: DBSCAN neighborhood radius in kilometers (default 50.0 km).
        min_samples: DBSCAN minimum samples per cluster core (default 5).
        threshold_percentile: Percentile threshold for elevated pixels (default 90.0).
        observation_period: Optional string describing the observation time window.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - clusters_df: Summary of detected hotspot clusters.
            - elevated_df: Elevated pixel points with assigned cluster IDs (-1 for noise).
            - report: Complete quality and clustering report.
    """
    # 1. Load data
    if isinstance(data, (str, Path)):
        csv_path = Path(data)
        if not csv_path.exists():
            raise FileNotFoundError(f"TROPOMI HCHO data file not found at: {csv_path}")
        logger.info("Loading TROPOMI data from: %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        df = data.copy()

    # Edge case: Empty input
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning 0 clusters.")
        empty_clusters = pd.DataFrame(
            columns=[
                "cluster_id",
                "centroid_lat",
                "centroid_lon",
                "pixel_count",
                "mean_hcho_molecules_cm2",
                "max_hcho_molecules_cm2",
                "min_hcho_molecules_cm2",
            ]
        )
        empty_elevated = pd.DataFrame(
            columns=["latitude", "longitude", "hcho_molecules_cm2", "cluster_id"]
        )
        empty_report = {
            "observation_period": observation_period or "Unknown",
            "total_pixels": 0,
            "elevated_pixels": 0,
            "num_clusters": 0,
            "noise_pixels": 0,
            "clustering_parameters": {
                "eps_km": eps_km,
                "eps_radians": eps_km / EARTH_RADIUS_KM,
                "min_samples": min_samples,
                "threshold_percentile": threshold_percentile,
                "threshold_value_molecules_cm2": 0.0,
            },
            "concentration_distribution_molecules_cm2": compute_concentration_distribution(
                pd.Series(dtype=float)
            ),
            "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        }
        return empty_clusters, empty_elevated, empty_report

    # Validate columns
    concentration_col, _ = validate_input_data(df)

    # Ensure hcho_molecules_cm2 column exists
    if "hcho_molecules_cm2" not in df.columns:
        df["hcho_molecules_cm2"] = df["hcho_mol_m2"] * MOL_M2_TO_MOLECULES_CM2

    # Drop invalid rows
    valid_df = df.dropna(subset=["latitude", "longitude", "hcho_molecules_cm2"]).copy()

    # 2. Report concentration distribution
    dist_stats = compute_concentration_distribution(valid_df["hcho_molecules_cm2"])
    logger.info("Concentration distribution (molecules/cm²): %s", dist_stats)

    # 3. Data-driven threshold (e.g. P90)
    threshold_value = float(
        np.percentile(valid_df["hcho_molecules_cm2"].to_numpy(), threshold_percentile)
    )
    logger.info(
        "Using P%.1f threshold: %.4e molecules/cm²",
        threshold_percentile,
        threshold_value,
    )

    elevated_mask = valid_df["hcho_molecules_cm2"] >= threshold_value
    elevated_df = valid_df[elevated_mask].copy()

    logger.info(
        "Identified %d elevated pixels (out of %d total pixels).",
        len(elevated_df),
        len(valid_df),
    )

    # If no pixels meet threshold or fewer than min_samples
    if elevated_df.empty or len(elevated_df) < min_samples:
        elevated_df["cluster_id"] = -1
        empty_clusters = pd.DataFrame(
            columns=[
                "cluster_id",
                "centroid_lat",
                "centroid_lon",
                "pixel_count",
                "mean_hcho_molecules_cm2",
                "max_hcho_molecules_cm2",
                "min_hcho_molecules_cm2",
            ]
        )
        report = {
            "observation_period": observation_period or "Unknown",
            "total_pixels": len(valid_df),
            "elevated_pixels": len(elevated_df),
            "num_clusters": 0,
            "noise_pixels": len(elevated_df),
            "clustering_parameters": {
                "eps_km": eps_km,
                "eps_radians": eps_km / EARTH_RADIUS_KM,
                "min_samples": min_samples,
                "threshold_percentile": threshold_percentile,
                "threshold_value_molecules_cm2": threshold_value,
            },
            "concentration_distribution_molecules_cm2": dist_stats,
            "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        }
        return empty_clusters, elevated_df, report

    # 4. DBSCAN clustering with Haversine distance
    # Convert lat/lon to radians. Order must be [latitude, longitude] for sklearn haversine metric.
    coords_rad = np.radians(elevated_df[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    cluster_labels = db.fit_predict(coords_rad)
    elevated_df["cluster_id"] = cluster_labels

    # 5. Compute cluster summaries (excluding noise label -1)
    unique_clusters = [c for c in sorted(np.unique(cluster_labels)) if c != -1]
    logger.info("DBSCAN identified %d distinct clusters.", len(unique_clusters))

    cluster_records: List[Dict[str, Any]] = []
    for c_id in unique_clusters:
        pts = elevated_df[elevated_df["cluster_id"] == c_id]
        c_lat = float(pts["latitude"].mean())
        c_lon = float(pts["longitude"].mean())
        cnt = int(len(pts))
        c_mean = float(pts["hcho_molecules_cm2"].mean())
        c_max = float(pts["hcho_molecules_cm2"].max())
        c_min = float(pts["hcho_molecules_cm2"].min())

        record: Dict[str, Any] = {
            "cluster_id": int(c_id),
            "centroid_lat": round(c_lat, 6),
            "centroid_lon": round(c_lon, 6),
            "pixel_count": cnt,
            "mean_hcho_molecules_cm2": c_mean,
            "max_hcho_molecules_cm2": c_max,
            "min_hcho_molecules_cm2": c_min,
        }

        # Include mol/m^2 if available in source
        if "hcho_mol_m2" in pts.columns:
            record["mean_hcho_mol_m2"] = float(pts["hcho_mol_m2"].mean())
            record["max_hcho_mol_m2"] = float(pts["hcho_mol_m2"].max())

        cluster_records.append(record)

    clusters_df = pd.DataFrame(cluster_records)
    noise_count = int((elevated_df["cluster_id"] == -1).sum())

    # 6. Quality & parameter report
    report = {
        "observation_period": observation_period or "Unknown",
        "total_pixels": len(valid_df),
        "elevated_pixels": len(elevated_df),
        "num_clusters": len(unique_clusters),
        "clustered_pixels": len(elevated_df) - noise_count,
        "noise_pixels": noise_count,
        "clustering_parameters": {
            "eps_km": eps_km,
            "eps_radians": eps_rad,
            "min_samples": min_samples,
            "threshold_percentile": threshold_percentile,
            "threshold_value_molecules_cm2": threshold_value,
            "distance_metric": "haversine",
            "earth_radius_km": EARTH_RADIUS_KM,
        },
        "concentration_distribution_molecules_cm2": dist_stats,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
    }

    return clusters_df, elevated_df, report


def export_hotspots_geojson(clusters_df: pd.DataFrame, output_path: Union[str, Path]) -> Dict[str, Any]:
    """Export hotspot cluster centroids as an RFC 7946 compliant GeoJSON FeatureCollection.

    Each feature represents a cluster centroid with geometry Point [longitude, latitude].

    Args:
        clusters_df: DataFrame containing cluster summaries.
        output_path: Path to write the GeoJSON file.

    Returns:
        Dict[str, Any]: The GeoJSON FeatureCollection dictionary.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []
    for _, row in clusters_df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["centroid_lon"]), float(row["centroid_lat"])],
            },
            "properties": {
                "cluster_id": int(row["cluster_id"]),
                "pixel_count": int(row["pixel_count"]),
                "centroid_lat": float(row["centroid_lat"]),
                "centroid_lon": float(row["centroid_lon"]),
                "mean_hcho_molecules_cm2": float(row["mean_hcho_molecules_cm2"]),
                "max_hcho_molecules_cm2": float(row["max_hcho_molecules_cm2"]),
                "min_hcho_molecules_cm2": float(row["min_hcho_molecules_cm2"]),
            },
        }
        if "mean_hcho_mol_m2" in row and not pd.isna(row["mean_hcho_mol_m2"]):
            feature["properties"]["mean_hcho_mol_m2"] = float(row["mean_hcho_mol_m2"])
            feature["properties"]["max_hcho_mol_m2"] = float(row["max_hcho_mol_m2"])

        features.append(feature)

    geojson_dict = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2)

    logger.info("Saved GeoJSON FeatureCollection with %d features to %s", len(features), out_file)
    return geojson_dict


def export_results(
    clusters_df: pd.DataFrame,
    report: Dict[str, Any],
    geojson_path: Union[str, Path] = "outputs/hotspots/hcho_hotspots.geojson",
    summary_path: Union[str, Path] = "outputs/hotspots/hcho_clusters_summary.csv",
    metrics_path: Union[str, Path] = "outputs/metrics/hcho_quality_report.json",
) -> None:
    """Export all clustering outputs: GeoJSON, summary CSV, and metrics JSON.

    Args:
        clusters_df: DataFrame with cluster summaries.
        report: Quality and parameter report dictionary.
        geojson_path: Destination path for GeoJSON hotspots.
        summary_path: Destination path for tabular cluster summary.
        metrics_path: Destination path for quality report JSON.
    """
    # 1. GeoJSON
    export_hotspots_geojson(clusters_df, geojson_path)

    # 2. Summary CSV
    sum_file = Path(summary_path)
    sum_file.parent.mkdir(parents=True, exist_ok=True)
    clusters_df.to_csv(sum_file, index=False)
    logger.info("Saved clusters summary CSV to %s", sum_file)

    # 3. Quality report JSON
    met_file = Path(metrics_path)
    met_file.parent.mkdir(parents=True, exist_ok=True)
    with open(met_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved quality report JSON to %s", met_file)


def run_hotspot_pipeline(
    input_path: Union[str, Path] = "data/processed/tropomi_hcho_india.csv",
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    geojson_path: Union[str, Path] = "outputs/hotspots/hcho_hotspots.geojson",
    summary_path: Union[str, Path] = "outputs/hotspots/hcho_clusters_summary.csv",
    metrics_path: Union[str, Path] = "outputs/metrics/hcho_quality_report.json",
    observation_period: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete HCHO hotspot detection and reporting workflow.

    Args:
        input_path: Path to processed TROPOMI CSV.
        eps_km: DBSCAN eps in kilometers.
        min_samples: DBSCAN minimum samples per cluster.
        threshold_percentile: Percentile threshold for elevated pixels.
        geojson_path: Output GeoJSON path.
        summary_path: Output summary CSV path.
        metrics_path: Output metrics JSON path.
        observation_period: Observation time window string.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Clusters summary DataFrame and quality report.
    """
    clusters_df, elevated_df, report = detect_hcho_hotspots(
        data=input_path,
        eps_km=eps_km,
        min_samples=min_samples,
        threshold_percentile=threshold_percentile,
        observation_period=observation_period,
    )

    export_results(
        clusters_df=clusters_df,
        report=report,
        geojson_path=geojson_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
    )

    return clusters_df, report


def main() -> None:
    """CLI entry point for HCHO Hotspot Detection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Detect and cluster HCHO atmospheric hotspots from TROPOMI data."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/processed/tropomi_hcho_india.csv",
        help="Path to processed TROPOMI HCHO CSV.",
    )
    parser.add_argument(
        "--eps-km",
        type=float,
        default=DEFAULT_EPS_KM,
        help="DBSCAN clustering radius in km (default 50.0).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="DBSCAN minimum samples per cluster (default 5).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENTILE,
        help="Percentile threshold for elevated pixels (default 90.0).",
    )
    parser.add_argument(
        "--geojson-out",
        type=str,
        default="outputs/hotspots/hcho_hotspots.geojson",
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/hotspots/hcho_clusters_summary.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="outputs/metrics/hcho_quality_report.json",
        help="Output metrics JSON path.",
    )

    args = parser.parse_args()

    clusters_df, report = run_hotspot_pipeline(
        input_path=args.input_csv,
        eps_km=args.eps_km,
        min_samples=args.min_samples,
        threshold_percentile=args.percentile,
        geojson_path=args.geojson_out,
        summary_path=args.summary_out,
        metrics_path=args.metrics_out,
    )

    print("\n=== HCHO Hotspot Detection Completed ===")
    print(f"Total Pixels Analyzed: {report['total_pixels']}")
    print(f"Elevated Pixels: {report['elevated_pixels']} (Threshold P{args.percentile}: {report['clustering_parameters']['threshold_value_molecules_cm2']:.4e} molec/cm²)")
    print(f"Hotspot Clusters Detected: {report['num_clusters']}")
    print(f"Outputs written to:")
    print(f"  - GeoJSON: {args.geojson_out}")
    print(f"  - Summary CSV: {args.summary_out}")
    print(f"  - Quality Metrics: {args.metrics_out}")
    print(f"\n[NOTE]: {report['scientific_disclaimer']}\n")


if __name__ == "__main__":
    main()
