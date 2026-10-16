"""TROPOMI Formaldehyde (HCHO) Data Acquisition Module.

This module acquires Level-3 tropospheric formaldehyde (HCHO) column density
data from the Sentinel-5P TROPOMI sensor via Google Earth Engine (GEE).

Scientific Details:
- Dataset: COPERNICUS/S5P/OFFL/L3_HCHO (Offline Level-3 HCHO)
- Primary Band: tropospheric_HCHO_column_number_density
- Cloud Mask: cloud_fraction < 0.3
- Spatial Coverage: India Bounding Box [68.0, 6.5, 97.5, 37.5] (min_lon, min_lat, max_lon, max_lat)
- Units:
    * Raw GEE Unit: mol/m^2 (tropospheric column number density)
    * Standard Unit: molecules/cm^2
    * Conversion Factor: 1 mol/m^2 = 6.02214e19 molecules/cm^2
      (Avogadro's constant 6.02214076e23 molecules/mol divided by 10,000 cm^2/m^2)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Set up module logger
logger = logging.getLogger(__name__)

# Constants
MOL_M2_TO_MOLECULES_CM2: float = 6.02214e19
INDIA_BBOX: List[float] = [68.0, 6.5, 97.5, 37.5]  # [min_lon, min_lat, max_lon, max_lat]
DEFAULT_START_DATE: str = "2024-05-01"
DEFAULT_END_DATE: str = "2024-05-31"
DEFAULT_CLOUD_THRESHOLD: float = 0.3
DEFAULT_SCALE_M: int = 10000
DEFAULT_MAX_PIXELS: int = 50000
DEFAULT_OUTPUT_PATH: str = "data/processed/tropomi_hcho_india.csv"
HCHO_COLLECTION_ID: str = "COPERNICUS/S5P/OFFL/L3_HCHO"
HCHO_BAND_NAME: str = "tropospheric_HCHO_column_number_density"
CLOUD_BAND_NAME: str = "cloud_fraction"


def init_gee(project: Optional[str] = None) -> bool:
    """Initialize Google Earth Engine API.

    Handles authentication errors gracefully without raising uncaught exceptions.

    Args:
        project: Optional Google Cloud Project ID for Earth Engine initialization.

    Returns:
        bool: True if initialized successfully, False otherwise.
    """
    try:
        import ee

        # Check for project override in environment if not provided
        ee_project = project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

        if ee_project:
            logger.info("Initializing Google Earth Engine with project: %s", ee_project)
            ee.Initialize(project=ee_project)
        else:
            logger.info("Initializing Google Earth Engine with default credentials...")
            ee.Initialize()

        logger.info("Google Earth Engine initialized successfully.")
        return True

    except ImportError:
        logger.error(
            "Earth Engine Python library ('earthengine-api') is not installed.\n"
            "Install it via: pip install earthengine-api"
        )
        return False
    except Exception as exc:
        logger.error(
            "Failed to initialize Google Earth Engine: %s\n"
            "====================================================================\n"
            "GEE AUTHENTICATION REQUIRED:\n"
            "1. Run 'earthengine authenticate' in your command shell.\n"
            "2. Ensure you have access to a Google Cloud Project with Earth Engine enabled.\n"
            "3. Optionally set the EE_PROJECT environment variable:\n"
            "   set EE_PROJECT=your-project-id\n"
            "====================================================================",
            exc,
        )
        return False


def test_gee_access(project: Optional[str] = None) -> bool:
    """Test GEE authentication and access to Sentinel-5P HCHO data.

    1. Tries to initialize Earth Engine.
    2. Loads a single day of S5P HCHO over India.
    3. Samples 5 random points from India.
    4. Prints and logs raw values with scientific units.
    5. Returns True if successful, False otherwise.

    Args:
        project: Optional Google Cloud Project ID.

    Returns:
        bool: True if test succeeded, False otherwise.
    """
    print("\n--- Testing Google Earth Engine Access for Sentinel-5P HCHO ---")
    if not init_gee(project=project):
        print("[FAIL] GEE authentication failed or credentials not configured.")
        return False

    try:
        import ee

        print("[OK] GEE initialized successfully.")
        test_start = "2024-05-15"
        test_end = "2024-05-16"
        bbox = INDIA_BBOX
        roi = ee.Geometry.Rectangle(bbox)

        print(f"Loading single-day HCHO data ({test_start} to {test_end}) over India...")
        collection = (
            ee.ImageCollection(HCHO_COLLECTION_ID)
            .filterBounds(roi)
            .filterDate(test_start, test_end)
            .select([HCHO_BAND_NAME, CLOUD_BAND_NAME])
        )

        collection_size = collection.size().getInfo()
        logger.info("Test date collection size: %d image(s)", collection_size)

        if collection_size == 0:
            print(f"[WARN] No images found for {test_start}. Querying broader 3-day window...")
            collection = (
                ee.ImageCollection(HCHO_COLLECTION_ID)
                .filterBounds(roi)
                .filterDate("2024-05-14", "2024-05-17")
                .select([HCHO_BAND_NAME, CLOUD_BAND_NAME])
            )

        # Cloud filter: cloud_fraction < 0.3
        def _filter_clouds(img: Any) -> Any:
            c = img.select(CLOUD_BAND_NAME)
            return img.updateMask(c.lt(DEFAULT_CLOUD_THRESHOLD))

        masked_img = (
            collection.map(_filter_clouds)
            .select(HCHO_BAND_NAME)
            .mean()
            .clip(roi)
        )

        # Add lat/lon bands for sampling
        latlon = ee.Image.pixelLonLat()
        sample_img = masked_img.addBands(latlon)

        # Sample 5 points
        sample_fc = sample_img.sample(
            region=roi,
            scale=DEFAULT_SCALE_M,
            numPixels=5,
            geometries=True,
        )

        features = sample_fc.getInfo().get("features", [])
        if not features:
            print("[WARN] No valid pixels returned from test sample (possibly heavy cloud cover).")
            return True

        print(f"[OK] Successfully sampled {len(features)} test points over India:")
        print(f"{'#':<3} {'Latitude':<12} {'Longitude':<12} {'Raw (mol/m²)':<20} {'Converted (molec/cm²)':<22}")
        print("-" * 72)

        for i, feat in enumerate(features, start=1):
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [None, None])
            lon = props.get("longitude", coords[0])
            lat = props.get("latitude", coords[1])
            val_mol = props.get(HCHO_BAND_NAME)

            if val_mol is not None:
                val_molec = float(val_mol) * MOL_M2_TO_MOLECULES_CM2
                print(
                    f"{i:<3} {lat:<12.4f} {lon:<12.4f} {val_mol:<20.4e} {val_molec:<22.4e}"
                )
            else:
                print(f"{i:<3} {lat} {lon} {'None':<20} {'None':<22}")

        print("------------------------------------------------------------------------")
        print("[SUCCESS] Sentinel-5P HCHO data acquisition verified.\n")
        return True

    except Exception as exc:
        logger.error("Error during GEE access test: %s", exc, exc_info=True)
        print(f"[FAIL] Error accessing S5P HCHO data: {exc}")
        return False


def build_quality_report(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    bbox: List[float],
) -> Dict[str, Any]:
    """Generate comprehensive data quality report for acquired HCHO data.

    Args:
        df: Pandas DataFrame containing acquired observations.
        start_date: Observation start date (YYYY-MM-DD).
        end_date: Observation end date (YYYY-MM-DD).
        bbox: Geospatial bounding box [min_lon, min_lat, max_lon, max_lat].

    Returns:
        Dict[str, Any]: Data quality statistics and metadata.
    """
    total_rows = len(df)
    null_count_mol = int(df["hcho_mol_m2"].isna().sum()) if "hcho_mol_m2" in df else 0
    null_count_molec = (
        int(df["hcho_molecules_cm2"].isna().sum())
        if "hcho_molecules_cm2" in df
        else 0
    )

    clean_series = df["hcho_molecules_cm2"].dropna() if "hcho_molecules_cm2" in df else pd.Series(dtype=float)

    if not clean_series.empty:
        percentiles = {
            "min": float(np.min(clean_series)),
            "max": float(np.max(clean_series)),
            "mean": float(np.mean(clean_series)),
            "median": float(np.median(clean_series)),
            "p25": float(np.percentile(clean_series, 25)),
            "p75": float(np.percentile(clean_series, 75)),
            "p90": float(np.percentile(clean_series, 90)),
            "p95": float(np.percentile(clean_series, 95)),
            "p99": float(np.percentile(clean_series, 99)),
        }
    else:
        percentiles = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    report = {
        "dataset": HCHO_COLLECTION_ID,
        "date_range": {"start_date": start_date, "end_date": end_date},
        "bounding_box": {
            "min_lon": bbox[0],
            "min_lat": bbox[1],
            "max_lon": bbox[2],
            "max_lat": bbox[3],
        },
        "row_count": total_rows,
        "valid_row_count": len(clean_series),
        "null_counts": {
            "hcho_mol_m2": null_count_mol,
            "hcho_molecules_cm2": null_count_molec,
        },
        "percentile_statistics_molecules_cm2": percentiles,
        "units": {
            "hcho_mol_m2": "mol/m^2",
            "hcho_molecules_cm2": "molecules/cm^2",
        },
        "conversion_factor": MOL_M2_TO_MOLECULES_CM2,
    }

    return report


def fetch_tropomi_hcho_composite(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    bbox: Optional[List[float]] = None,
    cloud_fraction_threshold: float = DEFAULT_CLOUD_THRESHOLD,
    scale_m: int = DEFAULT_SCALE_M,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    output_path: Optional[str] = DEFAULT_OUTPUT_PATH,
    project: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch monthly mean composite of TROPOMI HCHO over India from GEE.

    Workflow:
    1. Authenticates with Google Earth Engine.
    2. Loads Sentinel-5P OFFL L3 HCHO collection over the region.
    3. Filters date range and bounds.
    4. Applies cloud fraction < cloud_fraction_threshold mask.
    5. Selects tropospheric_HCHO_column_number_density band.
    6. Computes mean composite and clips to bounding box.
    7. Samples into a tabular DataFrame with columns: latitude, longitude, hcho_mol_m2.
    8. Converts to molecules/cm^2 (multiplies by 6.02214e19).
    9. Saves result to CSV if output_path is provided.
    10. Generates and returns a quality report.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]. Defaults to India bbox.
        cloud_fraction_threshold: Maximum cloud fraction threshold (default 0.3).
        scale_m: Sampling scale in meters (default 10,000m ~ 10km).
        max_pixels: Maximum number of pixels to sample (default 50,000).
        output_path: File path to save output CSV.
        project: Optional Google Cloud Project ID.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Sampled DataFrame and data quality report.

    Raises:
        RuntimeError: If GEE initialization fails or data cannot be retrieved.
    """
    bbox = bbox or INDIA_BBOX

    logger.info("Initializing GEE for TROPOMI HCHO acquisition...")
    if not init_gee(project=project):
        raise RuntimeError(
            "Google Earth Engine initialization failed. "
            "Please authenticate using 'earthengine authenticate' before running acquisition."
        )

    import ee

    logger.info(
        "Querying TROPOMI HCHO from %s to %s over bbox %s (cloud fraction < %.2f)...",
        start_date,
        end_date,
        bbox,
        cloud_fraction_threshold,
    )

    roi = ee.Geometry.Rectangle(bbox)
    collection = (
        ee.ImageCollection(HCHO_COLLECTION_ID)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .select([HCHO_BAND_NAME, CLOUD_BAND_NAME])
    )

    count = collection.size().getInfo()
    logger.info("Retrieved %d images in date range.", count)
    if count == 0:
        raise RuntimeError(
            f"No TROPOMI HCHO images found for date range {start_date} to {end_date} in bbox {bbox}."
        )

    # Cloud masking
    def _apply_cloud_mask(img: Any) -> Any:
        cloud = img.select(CLOUD_BAND_NAME)
        mask = cloud.lt(cloud_fraction_threshold)
        return img.updateMask(mask)

    logger.info("Computing mean composite and clipping to region...")
    composite = (
        collection.map(_apply_cloud_mask)
        .select(HCHO_BAND_NAME)
        .mean()
        .clip(roi)
    )

    # Add coordinates band for point extraction
    latlon = ee.Image.pixelLonLat()
    composite_with_coords = composite.addBands(latlon)

    logger.info(
        "Sampling composite image at scale %d m (maxPixels=%d)...",
        scale_m,
        max_pixels,
    )
    sampled = composite_with_coords.sample(
        region=roi,
        scale=scale_m,
        numPixels=max_pixels,
        geometries=True,
    )

    features = sampled.getInfo().get("features", [])
    logger.info("Extracted %d sampled points from GEE.", len(features))

    records: List[Dict[str, float]] = []
    for f in features:
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None])
        val = props.get(HCHO_BAND_NAME)
        lon = props.get("longitude", coords[0])
        lat = props.get("latitude", coords[1])

        if val is not None and lat is not None and lon is not None:
            records.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "hcho_mol_m2": float(val),
            })

    if not records:
        raise RuntimeError("Sampling returned 0 valid HCHO data points.")

    df = pd.DataFrame(records)

    # Convert to standard units: molecules/cm^2
    df["hcho_molecules_cm2"] = df["hcho_mol_m2"] * MOL_M2_TO_MOLECULES_CM2

    # Quality report
    quality_report = build_quality_report(
        df=df,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
    )

    # Save to CSV if requested
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        logger.info("Saved %d processed HCHO records to %s", len(df), out_file)

    return df, quality_report


def main() -> None:
    """Command-line interface for TROPOMI HCHO data acquisition."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Acquire Sentinel-5P TROPOMI HCHO composite data over India."
    )
    parser.add_argument(
        "--test-access",
        action="store_true",
        help="Run GEE connectivity and sample access test only.",
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
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud project ID for GEE.",
    )
    parser.add_argument(
        "--scale-m",
        type=int,
        default=DEFAULT_SCALE_M,
        help="Sampling scale in meters.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum pixels to sample.",
    )

    args = parser.parse_args()

    if args.test_access:
        success = test_gee_access(project=args.project)
        sys.exit(0 if success else 1)

    try:
        df, report = fetch_tropomi_hcho_composite(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
            project=args.project,
            scale_m=args.scale_m,
            max_pixels=args.max_pixels,
        )
        print("\n--- TROPOMI HCHO Acquisition Quality Report ---")
        print(f"Total Rows: {report['row_count']}")
        print(f"Date Range: {report['date_range']['start_date']} to {report['date_range']['end_date']}")
        print("Concentration Percentiles (molecules/cm²):")
        for k, v in report["percentile_statistics_molecules_cm2"].items():
            print(f"  {k.upper():<8}: {v:.4e}")
        print(f"Saved to: {args.output}")

    except Exception as exc:
        logger.error("Data acquisition failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
