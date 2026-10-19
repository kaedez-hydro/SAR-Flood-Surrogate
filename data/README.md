# Data Directory

This directory contains data files for the SAR-Flood-Surrogate project.

## Directory Structure

```
data/
├── README.md              # This file
├── sample_patches/        # Sample training patches (if included)
└── patch_info/           # Patch metadata and assignments
```

## Full Dataset

Due to the large file size (~60 GB), the complete training dataset is not included in this repository.

### Download Full Dataset

**Option 1: Zenodo** (Recommended)
- DOI: [To be assigned]
- URL: [To be provided]
- Size: ~60 GB
- Contents:
  - 1,347 training/validation patches
  - 6-8 input channels per patch
  - Binary flood labels from SAR
  - Patch assignment information

**Option 2: Request from Authors**
- Contact: [your.email@university.edu]
- Please specify your research purpose

### Dataset Description

**Patches**:
- Size: 256 × 256 pixels
- Resolution: 100m per pixel
- Geographic coverage: ~25.6 km × 25.6 km per patch
- Format: NumPy array (.npy)

**Training Set**:
- Total patches: 1,267
- Percentage: 94.06%
- Spatial coverage: Multiple subbasins

**Validation Set**:
- Total patches: 80
- Percentage: 5.94%
- Subbasins: Dongdian, Langouwa, Xiaoqinghe

**Input Channels** (6-channel baseline):
1. **DEM** - Digital Elevation Model (m)
2. **Slope** - Terrain gradient (degrees)
3. **Event Rainfall** - Cumulative rainfall during event (mm)
4. **Initial Water** - Pre-event water body indicator
5. **Soil Moisture** - Volumetric soil water content (m³/m³)
6. **API** - Antecedent Precipitation Index (15-day window)

**Optional Channels**:
7. **R1d** - 1-day antecedent rainfall (mm)
8. **P3d** - 3-day antecedent rainfall (mm)

**Labels**:
- Format: Binary mask (256 × 256)
- Values: 0 (non-flood), 1 (flood)
- Source: SAR composites (Sentinel-1)
- Period: July 28 - August 23, 2023

## Data Processing

### Original Data Sources

**SAR Imagery**:
- Source: Sentinel-1 C-band
- Repository: Copernicus Open Access Hub
- URL: https://scihub.copernicus.eu/
- Product type: GRD (Ground Range Detected)
- Polarization: VV + VH

**Meteorological Data**:
- Source: ERA5-Land reanalysis
- Repository: Copernicus Climate Data Store
- URL: https://cds.climate.copernicus.eu/
- Variables: Soil moisture, precipitation

**Terrain Data**:
- Source: SRTM DEM (30m)
- Repository: USGS EarthExplorer
- URL: https://earthexplorer.usgs.gov/
- Processing: Resampled to 100m

### Processing Pipeline

1. **SAR Processing**:
   - Radiometric calibration
   - Speckle filtering
   - Geocoding to UTM projection
   - Water body classification
   - Temporal compositing

2. **Terrain Processing**:
   - DEM resampling to 100m
   - Slope calculation
   - Reprojection to common grid

3. **Meteorological Processing**:
   - ERA5-Land download
   - Spatial interpolation
   - Temporal aggregation
   - API calculation

4. **Patch Generation**:
   - Sliding window extraction (256×256, stride 64)
   - Channel stacking
   - Train/validation split by subbasin
   - Quality control and filtering

## Sample Data

A small subset of sample patches is included in `sample_patches/` for testing and demonstration purposes.

## Patch Assignment

The file `patch_info/patch_assignments.csv` contains:
- Patch ID
- Row and column indices
- Subbasin name
- Split (train/val)
- Additional metadata

## Data Format

### Input Patches

**File naming**: `patch_{row}_{col}.npy`

**Array shape**: (256, 256, C) where C = 6, 7, or 8 channels

**Data type**: float32

**Example loading**:
```python
import numpy as np

# Load input patch
patch = np.load('patch_1024_1792.npy')
print(f"Shape: {patch.shape}")  # (256, 256, 6)

# Access individual channels
dem = patch[:, :, 0]
slope = patch[:, :, 1]
rainfall = patch[:, :, 2]
# etc.
```

### Label Patches

**File naming**: `label_{row}_{col}.npy`

**Array shape**: (256, 256)

**Data type**: uint8

**Values**: 0 or 1

**Example loading**:
```python
import numpy as np

# Load label
label = np.load('label_1024_1792.npy')
print(f"Shape: {label.shape}")  # (256, 256)
print(f"Flood pixels: {np.sum(label)}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{yourname2025sar,
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[Your Names]},
  journal={Under Review},
  year={2025}
}
```

## License

This dataset is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- Share: copy and redistribute the material
- Adapt: remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit

## Contact

For questions about the data or download issues:
- Email: [your.email@university.edu]
- GitHub Issues: [repository URL]/issues
