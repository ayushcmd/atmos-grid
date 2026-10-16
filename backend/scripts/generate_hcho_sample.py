"""Generate benchmark India HCHO observations based on published TROPOMI distributions for pre-monsoon season."""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Background tropospheric HCHO column number density over India:
# Published literature: ~0.8e-4 to 1.8e-4 mol/m² (0.5e16 to 1.1e16 molec/cm²)
# Hotspot areas (Delhi-NCR, Singrauli / Sonbhadra power/coal belt, Mumbai industrial):
# Elevated values: ~2.5e-4 to 4.5e-4 mol/m² (1.5e16 to 2.7e16 molec/cm²)

MOL_M2_TO_MOLECULES_CM2 = 6.02214e19

# 1. Background points across India (grid sample within bounding box)
lats_bg = np.random.uniform(8.0, 35.0, 250)
lons_bg = np.random.uniform(69.0, 95.0, 250)
hcho_bg = np.random.uniform(0.6e-4, 1.4e-4, 250)

# 2. Delhi-NCR Cluster (28.6°N, 77.2°E) - 15 points
lats_delhi = 28.6 + np.random.normal(0, 0.15, 15)
lons_delhi = 77.2 + np.random.normal(0, 0.15, 15)
hcho_delhi = np.random.uniform(2.8e-4, 4.2e-4, 15)

# 3. Singrauli / Eastern Coal belt Cluster (24.2°N, 82.6°E) - 12 points
lats_sing = 24.2 + np.random.normal(0, 0.18, 12)
lons_sing = 82.6 + np.random.normal(0, 0.18, 12)
hcho_sing = np.random.uniform(2.6e-4, 3.9e-4, 12)

# 4. Mumbai-Thane Industrial Cluster (19.1°N, 73.0°E) - 10 points
lats_mum = 19.1 + np.random.normal(0, 0.15, 10)
lons_mum = 73.0 + np.random.normal(0, 0.15, 10)
hcho_mum = np.random.uniform(2.4e-4, 3.5e-4, 10)

all_lats = np.concatenate([lats_bg, lats_delhi, lats_sing, lats_mum])
all_lons = np.concatenate([lons_bg, lons_delhi, lons_sing, lons_mum])
all_hcho_mol = np.concatenate([hcho_bg, hcho_delhi, hcho_sing, hcho_mum])
all_hcho_molec = all_hcho_mol * MOL_M2_TO_MOLECULES_CM2

df = pd.DataFrame({
    "latitude": np.round(all_lats, 4),
    "longitude": np.round(all_lons, 4),
    "hcho_mol_m2": all_hcho_mol,
    "hcho_molecules_cm2": all_hcho_molec,
})

out_path = Path("data/processed/tropomi_hcho_india_benchmark.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Generated benchmark HCHO dataset with {len(df)} points at {out_path}")
print(df.describe())
