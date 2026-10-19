# Data Directory

This directory contains metadata and download instructions for the SAR-Flood-Surrogate project.

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

## Full Dataset

Complete processed training data (~60 GB) are available separately due to file size constraints.

### Download from Zenodo

**DOI**: [To be added]

**Contents**:
- 1,347 training/validation patches (256 × 256 pixels each)
- 6 input channels per patch
- Binary flood labels from SAR
- Patch metadata

---

## Data Description

### Patch Specifications

- **Size**: 256 × 256 pixels per patch
- **Resolution**: 0.0008983° (~77-79 m × 100 m)
- **CRS**: EPSG:4326 (WGS84 geographic coordinates)
- **Format**: NumPy array (.npy)
- **Data type**: float32 (inputs), uint8 (labels)

### Dataset Split

**Training Set**:
- Patches: 1,267 (94.06%)
- Spatial coverage: Multiple subbasins excluding validation holdouts

**Validation Set**:
- Patches: 80 (5.94%)
- Spatial coverage: Three validation subbasins (Dongdian, Langouwa, Xiaoqinghe)
- Strategy: Strict spatial block holdout

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
- **Processing**: Computed using standard terrain analysis

### Channel 3: Event Rainfall

- **Source**: GSMaP v6
- **Original resolution**: 0.1°
- **Temporal coverage**: July 19-30, 2023
- **Processing**: Cumulative rainfall during event, spatially interpolated
- **Units**: mm
- **URL**: https://sharaku.eorc.jaxa.jp/GSMaP/

### Channel 4: Initial Water Depth

- **Description**: Pre-event water body indicator
- **Source**: Derived from pre-event SAR or optical imagery
- **Values**: Binary or continuous depth estimate
- **Purpose**: Identifies permanent water bodies

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
- **Purpose**: Characterizes pre-event wetness

---

## Optional Channels (7/8-channel variants)

### Channel 7: R1d (1-day Antecedent Rainfall)

- **Source**: GSMaP v6
- **Period**: 1 day before event
- **Units**: mm
- **Note**: Adding this channel degraded performance (see Table 5)

### Channel 8: P3d (3-day Antecedent Rainfall)

- **Source**: GSMaP v6
- **Period**: 3 days before event
- **Units**: mm
- **Note**: Only tested with 8-channel Vanilla configuration

---

## Labels

### SAR-derived Flood Masks

- **Source**: Sentinel-1 C-band SAR
- **Repository**: Copernicus Data Space
- **URL**: https://dataspace.copernicus.eu/
- **Product**: GRD (Ground Range Detected)
- **Polarization**: VV + VH
- **Temporal compositing**: July 28 - August 23, 2023
- **Processing**:
  - Radiometric calibration
  - Speckle filtering (Lee sigma filter)
  - Geometric terrain correction
  - Water body classification (threshold-based)
  - Temporal compositing (maximum extent)

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
  - North-South: ~77-79 m (varies slightly with latitude)
  - East-West: ~100 m (at study area latitude ~39°N)
- **Full domain**: 2,688 rows × 5,248 columns

**Important**: This is a **geographic coordinate system**, not a projected UTM grid. Pixel area varies with latitude.

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

# Load input patch
input_patch = np.load(f'path/to/patches/input_{row}_{col}.npy')
print(f"Input shape: {input_patch.shape}")  # (256, 256, 6)

# Load label patch
label_patch = np.load(f'path/to/patches/label_{row}_{col}.npy')
print(f"Label shape: {label_patch.shape}")  # (256, 256)
print(f"Flood pixels: {np.sum(label_patch)}")

# Access individual channels
dem = input_patch[:, :, 0]
slope = input_patch[:, :, 1]
rainfall = input_patch[:, :, 2]
initial_water = input_patch[:, :, 3]
soil_moisture = input_patch[:, :, 4]
api = input_patch[:, :, 5]
```

---

## Data Processing Pipeline

### 1. Source Data Acquisition

- Download Sentinel-1 SAR scenes
- Download GSMaP rainfall grids
- Download SMAP soil moisture grids
- Download SRTM DEM tiles

### 2. Preprocessing

- SAR: Calibration → Filtering → Classification → Compositing
- Rainfall: Spatial interpolation → Temporal aggregation
- Soil Moisture: Temporal averaging → Spatial interpolation
- Terrain: Resampling → Slope calculation

### 3. Grid Harmonization

- Reproject all layers to EPSG:4326
- Resample to 0.0008983° resolution
- Crop to study domain extent
- Stack into multi-channel arrays

### 4. Patch Extraction

- Sliding window: 256 × 256 pixels
- Stride: 64 pixels
- Overlap: 192 pixels (75%)
- Total patches: 1,347

### 5. Spatial Split

- Identify validation subbasins
- Assign patches to train/val based on subbasin
- Generate patch_assignments.csv

---

## Data Quality Notes

### Coverage Overlap

Due to sliding-window extraction, **substantial pixel overlap** exists between patches:

- De-duplicated validation pixels: 1,105,920
- Covered by training patches: 790,528 (71.48%)
- Uncovered by training: 315,392 (28.52%)

See supplementary analysis (Table S1) for detailed coverage metrics.

### SAR Detection Limitations

- **Vegetation masking**: Dense vegetation may obscure water detection
- **Urban shadows**: Building shadows can be misclassified
- **Wet surfaces**: Very shallow water or saturated soil may not be detected
- **Temporal compositing**: Maximum extent may include transient features

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{[TO_BE_ADDED],
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[TO_BE_ADDED]},
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

Processed data are released for **non-commercial research purposes**.

Source data licenses:
- Sentinel-1: Free and open (Copernicus data policy)
- GSMaP: Free for research (JAXA policy)
- SMAP: Free and open (NASA EOSDIS policy)
- SRTM: Public domain (USGS)

---

## Contact

For questions about data access or processing:
- GitHub Issues: https://github.com/kaedez-hydro/SAR-Flood-Surrogate/issues
- Email: [TO_BE_ADDED]
