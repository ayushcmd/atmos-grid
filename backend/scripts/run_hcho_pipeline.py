"""AtmosGrid HCHO Hotspot Detection Pipeline Runner.

This script coordinates the end-to-end HCHO pipeline:
1. Pre-flight check: Tests Google Earth Engine authentication and access.
2. Stage 1: TROPOMI Sentinel-5P HCHO data acquisition and monthly mean composite generation.
3. Stage 2: Data-driven thresholding, Haversine DBSCAN spatial clustering, and GeoJSON export.
4. Reporting: Generates summary metrics, cluster catalog, and data quality reports.

Usage:
    .venv\\Scripts\\python backend\\scripts\\run_hcho_pipeline.py
    .venv\\Scripts\\python backend\\scripts\\run_hcho_pipeline.py --test-access-only
    .venv\\Scripts\\python backend\\scripts\\run_hcho_pipeline.py --input-csv data/processed/tropomi_hcho_india.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so backend imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.data.fetch_tropomi_hcho import (
    DEFAULT_CLOUD_THRESHOLD,
    DEFAULT_END_DATE,
    DEFAULT_MAX_PIXELS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SCALE_M,
    DEFAULT_START_DATE,
    INDIA_BBOX,
    fetch_tropomi_hcho_composite,
    test_gee_access,
)
from backend.src.models.hcho_hotspots import (
    DEFAULT_EPS_KM,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_THRESHOLD_PERCENTILE,
    detect_hcho_hotspots,
    export_results,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hcho_pipeline")


def run_pipeline(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    cloud_threshold: float = DEFAULT_CLOUD_THRESHOLD,
    scale_m: int = DEFAULT_SCALE_M,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    processed_csv: str = DEFAULT_OUTPUT_PATH,
    geojson_out: str = "outputs/hotspots/hcho_hotspots.geojson",
    summary_out: str = "outputs/hotspots/hcho_clusters_summary.csv",
    metrics_out: str = "outputs/metrics/hcho_quality_report.json",
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    skip_acquisition: bool = False,
    project: str | None = None,
) -> bool:
    """Execute the end-to-end HCHO hotspot detection pipeline.

    Args:
        start_date: Observation start date (YYYY-MM-DD).
        end_date: Observation end date (YYYY-MM-DD).
        cloud_threshold: Max cloud fraction for pixel inclusion.
        scale_m: Earth Engine sampling resolution in meters.
        max_pixels: Maximum number of pixels to sample.
        processed_csv: Path to saved/input processed CSV.
        geojson_out: Path to output GeoJSON feature collection.
        summary_out: Path to output tabular cluster summary CSV.
        metrics_out: Path to output quality report JSON.
        eps_km: DBSCAN clustering neighborhood distance in km.
        min_samples: DBSCAN minimum points per cluster core.
        percentile: Threshold percentile for elevated concentration pixels.
        skip_acquisition: If True, bypass GEE acquisition and use existing CSV.
        project: Google Cloud project ID for GEE.

    Returns:
        bool: True if pipeline completed successfully, False otherwise.
    """
    print("\n" + "=" * 75)
    print("       ATMOSGRID: HCHO HOTSPOT DETECTION PIPELINE")
    print("=" * 75)
    print(f"Observation Window : {start_date} to {end_date}")
    print(f"Region of Interest : India Bounding Box {INDIA_BBOX}")
    print(f"Clustering Params  : eps={eps_km} km, min_samples={min_samples}, threshold=P{percentile}")
    print("=" * 75 + "\n")

    csv_file = Path(processed_csv)
    gee_available = False

    # Step 1: Pre-flight GEE Access Test
    if not skip_acquisition:
        print("[Step 1/3] Testing Google Earth Engine Access...")
        gee_available = test_gee_access(project=project)
        if not gee_available:
            print("\n" + "!" * 75)
            print("WARNING: Google Earth Engine authentication is not configured.")
            print("To enable satellite acquisition:")
            print("  1. Run 'earthengine authenticate' in your command prompt.")
            print("  2. If using a specific GCP project, set EE_PROJECT=<project-id>.")
            print("!" * 75 + "\n")

            if csv_file.exists():
                print(f"[FALLBACK] Found existing processed dataset at: {csv_file}")
                print("Proceeding directly to Stage 2: Hotspot Clustering using existing data.\n")
            else:
                print(f"[ERROR] No existing processed dataset found at: {csv_file}")
                print("Pipeline cannot proceed without either GEE access or pre-processed data.")
                print("Exiting pipeline.\n")
                return False

    # Step 2: Data Acquisition (if GEE is available and not skipped)
    if not skip_acquisition and gee_available:
        print("[Step 2/3] Stage 1: Sentinel-5P TROPOMI HCHO Data Acquisition...")
        try:
            df, acq_report = fetch_tropomi_hcho_composite(
                start_date=start_date,
                end_date=end_date,
                bbox=INDIA_BBOX,
                cloud_fraction_threshold=cloud_threshold,
                scale_m=scale_m,
                max_pixels=max_pixels,
                output_path=str(csv_file),
                project=project,
            )
            print(f"[OK] Successfully acquired {len(df)} sampled points.")
            print(f"[OK] Processed CSV saved to: {csv_file}")
            print(
                f"[INFO] HCHO Mean: {acq_report['percentile_statistics_molecules_cm2']['mean']:.4e} molecules/cm²"
            )
        except Exception as exc:
            logger.error("Stage 1 data acquisition failed: %s", exc, exc_info=True)
            print(f"[FAIL] Data acquisition error: {exc}")
            if csv_file.exists():
                print(f"[FALLBACK] Falling back to existing processed CSV at: {csv_file}")
            else:
                return False
    else:
        print(f"[Step 2/3] Skipping acquisition; loading existing CSV: {csv_file}")

    # Step 3: Hotspot Detection and Clustering
    print("\n[Step 3/3] Stage 2: Elevated HCHO Detection & Spatial Clustering...")
    try:
        obs_window = f"{start_date} to {end_date}"
        clusters_df, elevated_df, report = detect_hcho_hotspots(
            data=csv_file,
            eps_km=eps_km,
            min_samples=min_samples,
            threshold_percentile=percentile,
            observation_period=obs_window,
        )

        # Export outputs
        export_results(
            clusters_df=clusters_df,
            report=report,
            geojson_path=geojson_out,
            summary_path=summary_out,
            metrics_path=metrics_out,
        )

        print("\n" + "=" * 75)
        print("                 PIPELINE EXECUTION SUMMARY")
        print("=" * 75)
        print(f"Total Pixels Analyzed : {report['total_pixels']}")
        print(
            f"Elevated Pixels       : {report['elevated_pixels']} (Threshold P{percentile} = {report['clustering_parameters']['threshold_value_molecules_cm2']:.4e} molec/cm²)"
        )
        print(f"Clusters Detected     : {report['num_clusters']}")
        print(f"Clustered Pixels      : {report.get('clustered_pixels', 0)}")
        print(f"Noise / Isolated Pts  : {report['noise_pixels']}")
        print("-" * 75)
        print("Concentration Distribution (molecules/cm²):")
        dist = report["concentration_distribution_molecules_cm2"]
        for stat in ["min", "median", "mean", "p75", "p90", "p95", "p99", "max"]:
            print(f"  {stat.upper():<8}: {dist.get(stat, 0.0):.4e}")
        print("-" * 75)
        print("Generated Artifacts:")
        print(f"  1. GeoJSON Hotspots   : {geojson_out}")
        print(f"  2. Cluster Summary    : {summary_out}")
        print(f"  3. Quality Report     : {metrics_out}")
        print("=" * 75)
        print(f"\n[NOTE]: {report['scientific_disclaimer']}\n")
        return True

    except Exception as exc:
        logger.error("Stage 2 Hotspot Clustering failed: %s", exc, exc_info=True)
        print(f"[FAIL] Hotspot clustering error: {exc}")
        return False


def main() -> None:
    """CLI entry point for running the HCHO pipeline."""
    parser = argparse.ArgumentParser(
        description="Run AtmosGrid TROPOMI HCHO Hotspot Detection Pipeline."
    )
    parser.add_argument(
        "--test-access-only",
        action="store_true",
        help="Only test Google Earth Engine access and exit.",
    )
    parser.add_argument(
        "--skip-acquisition",
        action="store_true",
        help="Skip GEE acquisition and run clustering directly on existing CSV.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Observation start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help="Observation end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to processed CSV input/output.",
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
        help="DBSCAN minimum points per cluster (default 5).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENTILE,
        help="Threshold percentile for elevated pixels (default 90.0).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud Project ID for Earth Engine.",
    )
    parser.add_argument(
        "--geojson-out",
        type=str,
        default="outputs/hotspots/hcho_hotspots.geojson",
        help="Destination path for GeoJSON output.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/hotspots/hcho_clusters_summary.csv",
        help="Destination path for summary CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="outputs/metrics/hcho_quality_report.json",
        help="Destination path for quality report JSON.",
    )

    args = parser.parse_args()

    if args.test_access_only:
        success = test_gee_access(project=args.project)
        sys.exit(0 if success else 1)

    success = run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        processed_csv=args.input_csv,
        geojson_out=args.geojson_out,
        summary_out=args.summary_out,
        metrics_out=args.metrics_out,
        eps_km=args.eps_km,
        min_samples=args.min_samples,
        percentile=args.percentile,
        skip_acquisition=args.skip_acquisition,
        project=args.project,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
