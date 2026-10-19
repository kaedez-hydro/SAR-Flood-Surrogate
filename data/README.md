# Data Directory

This directory contains metadata for the SAR-Flood-Surrogate project.

## Available Files

### patch_assignments.csv

**Description**: Training/validation split assignments for 1,347 patches

**Columns**:
- `patch_id`: Unique patch identifier
- `row_start`, `col_start`: Top-left corner in full domain grid
- `split`: 'train' (1,267 patches) or 'val' (80 patches)
- `subbasin`: Geographic subbasin identifier

**Usage**:
```python
import pandas as pd

assignments = pd.read_csv('data/patch_assignments.csv')
train_patches = assignments[assignments['split'] == 'train']
val_patches = assignments[assignments['split'] == 'val']

print(f"Training patches: {len(train_patches)}")
print(f"Validation patches: {len(val_patches)}")
```

---

## Data Description

### Patch Specifications

- **Size**: 256 × 256 pixels per patch
- **Resolution**: 0.0008983° (~76.7-78.8 m east-west, 99.7 m north-south)
- **CRS**: EPSG:4326 (WGS84 geographic coordinates)
- **Format**: NumPy array (.npy), channel-first (C, H, W)
- **Data type**: float32 (inputs), uint8 (labels)

### Dataset Split

**Training Set**:
- Patches: 1,267 (94.06%)
- Strategy: Spatial block constraint

**Validation Set**:
- Patches: 80 (5.94%)
- Validation subbasins: Dongdian, Langouwa, Xiaoqinghe
- Strategy: Strict spatial holdout

**Important**: Due to sliding-window extraction (256×256 patches, 64-pixel stride), substantial pixel overlap exists between patches. See supplementary analysis for coverage details.

---

## Input Channels (6-channel baseline)

### Channel 1: DEM (Digital Elevation Model)

- **Source**: SRTM GL1 v003
- **Original resolution**: 30m
- **Processing**: Resampled to 0.0008983°
- **Units**: meters above sea level
- **URL**: https://lpdaac.usgs.gov/products/srtmgl1v003/

### Channel 2: Slope

- **Derived from**: DEM
- **Calculation**: Terrain gradient in degrees
- **Units**: degrees (0-90)

### Channel 3: Event Rainfall

- **Source**: GSMaP v6
- **Original resolution**: 0.1°
- **Temporal coverage**: July 29 - August 2, 2023
- **Processing**: Cumulative rainfall during event, spatially interpolated
- **Units**: mm
- **URL**: https://sharaku.eorc.jaxa.jp/GSMaP/

### Channel 4: W₀ (Initial Water)

- **Description**: Pre-event water body binary mask
- **Source**: Single pre-event Sentinel-1 scene
- **Processing**: Classified by GNDPI + texture + random forest, thresholded at 0.5
- **Values**: Binary (0 or 1)

### Channel 5: Soil Moisture

- **Source**: SMAP Level-4 SPL4SMGP v007
- **Variable**: Surface soil moisture (0-5 cm layer)
- **Original resolution**: 9 km
- **Temporal**: Pre-event average
- **Units**: m³/m³ (volumetric)
- **URL**: https://nsidc.org/data/spl4smgp/

### Channel 6: API (Antecedent Precipitation Index)

- **Source**: Computed from GSMaP v6 rainfall
- **Formula**: API(t) = Σ[τ=1 to K] λ^τ × P(t-τ)
- **Parameters**: K=15 days, λ=0.8
- **Units**: mm (effective)

---

## Optional Channels (7-channel variant)

### Channel 7: R1d (1-day Antecedent Rainfall)

- **Source**: GSMaP v6
- **Period**: 1 day before event
- **Units**: mm
- **Note**: Adding this channel degraded performance (see Table 5)

---

## Labels

### SAR-derived Flood Masks

- **Source**: Sentinel-1 C-band SAR
- **Repository**: Copernicus Data Space
- **URL**: https://dataspace.copernicus.eu/
- **Temporal coverage**: July 28 - August 23, 2023
- **Processing**:
  - GNDPI + texture features
  - Random forest classification (Zhu et al., 2026)
  - Temporal compositing (union of all detection instances)

**Format**:
- Array shape: (256, 256)
- Data type: uint8
- Values: 0 (non-flood), 1 (flood)

---

## Grid Specification

All data layers are resampled to a common geographic grid:

- **CRS**: EPSG:4326 (WGS84)
- **Resolution**: 0.0008983° in both longitude and latitude
- **Approximate metric size**: 
  - East-West: ~76.7-78.8 m (varies slightly with longitude)
  - North-South: ~99.7 m
- **Source grid**: 2,199 rows × 4,685 columns
- **Patch coverage domain**: 2,176 rows × 4,672 columns
- **Top-left corner**: 113.5621°E, 40.0530°N

**Important**: This is a **geographic coordinate system**, not a projected grid. Pixel area varies with latitude.

---

## Data Loading Example

```python
import numpy as np
import pandas as pd

# Load patch assignments
assignments = pd.read_csv('data/patch_assignments.csv')

# Get first training patch
first_train = assignments[assignments['split'] == 'train'].iloc[0]
row, col = first_train['row_start'], first_train['col_start']

# Load input patch - channel-first format
input_patch = np.load(f'path/to/patches/input_{row}_{col}.npy')
print(f"Input shape: {input_patch.shape}")  # (6, 256, 256)

# Load label patch
label_patch = np.load(f'path/to/patches/label_{row}_{col}.npy')
print(f"Label shape: {label_patch.shape}")  # (256, 256)
print(f"Flood pixels: {np.sum(label_patch)}")

# Access individual channels (channel-first)
dem = input_patch[0, :, :]
slope = input_patch[1, :, :]
rainfall = input_patch[2, :, :]
initial_water = input_patch[3, :, :]
soil_moisture = input_patch[4, :, :]
api = input_patch[5, :, :]
```

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{zhu2025sar,
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={Zhu, Changjun and Zhang, Ting and Zhou, Keke and Li, Jianzhu and Kang, Aiqing and Ma, Qiang},
  journal={Under Review},
  year={2025}
}
```

And acknowledge the source data providers:
- Sentinel-1: ESA Copernicus Programme
- GSMaP: JAXA EORC
- SMAP: NASA NSIDC
- SRTM: NASA/USGS LPDAAC

---

## License

- **Code**: MIT License
- **Model weights and processed data**: CC BY 4.0

---

## Contact

For questions about data:
- GitHub Issues: https://github.com/kaedez-hydro/SAR-Flood-Surrogate/issues
