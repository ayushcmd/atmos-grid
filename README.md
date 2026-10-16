# AtmosGrid — Satellite-Based Surface AQI Mapping and HCHO Hotspot Detection over India

## Project Overview

AtmosGrid is a geospatial environmental analytics platform with two analytical pipelines:

1. **Surface AQI Estimation** — Estimate ground-level air quality using CPCB ground measurements, satellite observations, and meteorological features.
2. **HCHO Hotspot Detection** — Identify spatial clusters of elevated atmospheric formaldehyde concentrations from Sentinel-5P/TROPOMI observations.

## Quick Start

### Prerequisites
- Python 3.14+ (tested with 3.14.3)
- Google Earth Engine credentials (for TROPOMI HCHO pipeline)

### Setup
```bash
cd c:\Projects\AtmosGrid
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

### Run HCHO Hotspot Pipeline
```bash
python backend\scripts\run_hcho_pipeline.py
```

### Run CPCB Data Acquisition
```bash
python backend\scripts\run_cpcb_acquisition.py
```

### Run Local Streamlit Dashboard (Phase 1.5)
```bash
.venv\Scripts\streamlit run dashboard\app.py
```

### Run Tests
```bash
python -m pytest backend\tests\ -v
```

## Data Sources

| Dataset | Source | Auth |
|:--------|:-------|:-----|
| CPCB Ground AQI | OpenAQ AWS S3 Archive | None required |
| TROPOMI HCHO | Google Earth Engine | GEE credentials |
| ERA5 Meteorology | Copernicus CDS | CDS account (deferred) |

## Repository Structure

```
backend/
  src/
    data/       # Data acquisition and preprocessing
    models/     # ML models and analytics
  scripts/      # Pipeline entry points
  tests/        # Unit and integration tests
  requirements.txt
data/
  raw/          # Downloaded raw data (not in git)
  processed/    # Cleaned/processed data
outputs/
  predictions/  # AQI model outputs
  hotspots/     # HCHO cluster GeoJSON
  metrics/      # Evaluation metrics
models/         # Trained model artifacts
```

## License
Research and educational use.
