# SAR-Constrained Flood Inundation Surrogate Modelling

Deep learning surrogate model for rapid flood extent mapping using SAR observations and physical predictors.

[![DOI](https://zenodo.org/badge/DOI/[TO_BE_ADDED].svg)](https://doi.org/[TO_BE_ADDED])
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Paper

**Title**: SAR-constrained flood inundation surrogate modelling with physical predictors

**Study Event**: July 2023 Daqinghe River flood, North China Plain

**Status**: Under review

## Overview

This repository contains code, data, and results for training and evaluating deep learning surrogate models that predict flood inundation extent using terrain, meteorological, and hydrological inputs, supervised by Sentinel-1 SAR observations.

### Key Features

- Vanilla U-Net and ResNet34 U-Net architectures
- Physical predictor inputs: DEM, slope, rainfall, soil moisture, antecedent precipitation
- Spatial block validation with explicit coverage analysis
- Comprehensive performance evaluation including training-covered vs uncovered subsets

### Study Area

- **Basin**: Daqinghe River, North China Plain
- **Event**: July 2023 extreme flood
- **Domain**: 2,688 × 5,248 pixels
- **Resolution**: 0.0008983° (~77-79m × 100m, EPSG:4326)

## Quick Start

### Installation

```bash
git clone https://github.com/kaedez-hydro/SAR-Flood-Surrogate.git
cd SAR-Flood-Surrogate

conda create -n flood python=3.8
conda activate flood
pip install -r requirements.txt
```

### Download Pre-trained Models

Pre-trained model checkpoints (330 MB total) are available at:

**Zenodo**: [DOI to be added]

Download and place in `models/checkpoints/`.

### Run Inference

```python
import torch
import numpy as np
from code.models import VanillaUNet

# Load model
model = VanillaUNet(in_channels=6, out_channels=1)
checkpoint = torch.load('models/checkpoints/van_050105_6.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load input (H, W, 6)
input_data = np.load('your_patch.npy')
input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0).float()

# Inference
with torch.no_grad():
    output = torch.sigmoid(model(input_tensor))
    prediction = (output > 0.5).cpu().numpy()[0, 0]
```

## Results

### Table 5: Model Performance on 80 Validation Patches

| Model | Channels | IoU | F1 | Precision | Recall |
|-------|----------|-----|----|-----------| -------|
| ResNet34 U-Net | 6 | 0.400 | 0.572 | 0.745 | 0.464 |
| **Vanilla U-Net** | **6** | **0.464** | **0.634** | **0.817** | **0.517** |
| Vanilla U-Net | 7 | 0.364 | 0.533 | 0.644 | 0.455 |

### Performance by Training Coverage (Table S1)

De-duplicated validation pixels analysis:

| Subset | IoU | F1 | Precision | Recall | Support |
|--------|-----|----|-----------| -------|---------|
| Training-covered | 0.61 | 0.76 | 0.82 | 0.70 | 6,444 |
| Training-uncovered | 0.11 | 0.20 | 0.79 | 0.11 | 365 |
| All validation | 0.58 | 0.73 | 0.82 | 0.66 | 6,809 |

**Key Finding**: Performance in training-uncovered pixels (IoU=0.11) is substantially lower than training-covered areas (IoU=0.61), indicating limited spatial extrapolation.

## Repository Structure

```
SAR-Flood-Surrogate/
├── code/
│   ├── dataset.py              # Data loading and preprocessing
│   ├── models.py               # Model architectures
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils/
│       ├── make_spatial_split.py
│       └── run_supplementary_analysis.py
│
├── models/
│   └── checkpoints/            # Pre-trained weights (download separately)
│
├── data/
│   ├── patch_assignments.csv   # Training/validation split (1,267 / 80)
│   └── README.md              # Data description
│
├── results/
│   ├── evaluation/            # Evaluation metrics
│   └── supplementary_analysis/ # Coverage and bootstrap results
│
└── docs/
    └── figures/               # Supplementary figures S1-S4
```

## Training Configuration

All models were trained with:

- **Optimizer**: Adam
- **Learning rate**: 3×10⁻⁴
- **Learning rate schedule**: CosineAnnealingLR
- **Batch size**: 135 (training), 70 (validation)
- **Epochs**: 30,000 (with early stopping)
- **Early stopping patience**: 1,000 epochs
- **Loss function**: Weighted BCE (positive class weight: 4.0)
- **Weight decay**: 1×10⁻⁴
- **Data augmentation**: As described in supplementary materials

## Input Data Sources

### Satellite Observations

**SAR Flood Labels**:
- Source: Sentinel-1 C-band SAR
- Repository: Copernicus Data Space
- URL: https://dataspace.copernicus.eu/
- Period: July 28 - August 23, 2023
- Processing: Radiometric calibration, speckle filtering, water classification

### Physical Predictors

**Precipitation**:
- Source: GSMaP v6 (Global Satellite Mapping of Precipitation)
- Resolution: 0.1° (~10 km)
- Temporal: Hourly
- URL: https://sharaku.eorc.jaxa.jp/GSMaP/

**Soil Moisture**:
- Source: SMAP Level-4 SPL4SMGP v007
- Resolution: 9 km
- Temporal: 3-hourly
- URL: https://nsidc.org/data/spl4smgp/

**Terrain**:
- Source: SRTM GL1 v003 (30m DEM)
- Processing: Slope calculation, resampling
- URL: https://lpdaac.usgs.gov/products/srtmgl1v003/

## Grid Specification

All data are resampled to a common grid:

- **CRS**: EPSG:4326 (WGS84 geographic coordinates)
- **Resolution**: 0.0008983° (longitude and latitude)
- **Approximate metric**: ~77-79 m (north-south) × ~100 m (east-west) at study latitude
- **Domain extent**: 2,688 rows × 5,248 columns
- **Geographic bounds**: [To be specified from metadata]

**Note**: This is a geographic coordinate system, not a projected UTM grid. Pixel area varies slightly with latitude.

## Reproduce Results

### Reproduce Table 5

```bash
python code/evaluate.py \
    --checkpoint models/checkpoints/van_050105_6.pth \
    --data_split data/patch_assignments.csv \
    --output results/table5_reproduction.csv
```

### Reproduce Table S1 (Coverage Analysis)

```bash
python code/utils/run_supplementary_analysis.py \
    --mode coverage \
    --output results/supplementary_analysis/
```

## Data Availability

**Code**: https://github.com/kaedez-hydro/SAR-Flood-Surrogate

**Model Checkpoints and Processed Data**: Zenodo [DOI to be added]
- Pre-trained models (330 MB)
- Patch assignment file
- Validation predictions

**Source Data**:
- GSMaP v6: https://sharaku.eorc.jaxa.jp/GSMaP/
- SMAP SPL4SMGP: https://nsidc.org/data/spl4smgp/
- SRTM GL1: https://lpdaac.usgs.gov/products/srtmgl1v003/
- Sentinel-1: https://dataspace.copernicus.eu/

## Citation

If you use this code or data, please cite:

```bibtex
@article{[TO_BE_ADDED],
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[TO_BE_ADDED]},
  journal={Under Review},
  year={2025},
  doi={[TO_BE_ADDED]}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

Code and model weights are released under MIT License. Processed data are available for non-commercial research purposes.

## Contact

For questions about code, data, or methods:
- GitHub Issues: https://github.com/kaedez-hydro/SAR-Flood-Surrogate/issues
- Email: [TO_BE_ADDED]

## Acknowledgments

- Sentinel-1 SAR: ESA Copernicus Programme
- GSMaP: JAXA Earth Observation Research Center
- SMAP: NASA National Snow and Ice Data Center
- SRTM: NASA/USGS Land Processes Distributed Active Archive Center
