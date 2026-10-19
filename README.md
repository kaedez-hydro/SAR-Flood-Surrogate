# SAR-Constrained Flood Inundation Surrogate Modelling

Deep learning surrogate model for rapid flood extent mapping using SAR observations and physical predictors.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23465452.svg)](https://doi.org/10.5281/zenodo.23465452)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Paper

**Title**: SAR-constrained flood inundation surrogate modelling with physical predictors

**Study Event**: July 2023 Daqinghe River flood, North China Plain

**Status**: Under review

## Overview

This repository contains code, data, and results for training and evaluating deep learning surrogate models that predict flood inundation extent using terrain, meteorological, and hydrological inputs, supervised by Sentinel-1 SAR observations.

### Study Area

- **Basin**: Daqinghe River, North China Plain
- **Event rainfall**: July 29 - August 2, 2023
- **SAR monitoring**: July 28 - August 23, 2023
- **Source grid**: 2,199 rows × 4,685 columns
- **Patch coverage domain**: 2,176 rows × 4,672 columns
- **Top-left corner**: 113.5621°E, 40.0530°N
- **CRS**: EPSG:4326 (WGS84 geographic coordinates)
- **Resolution**: 0.0008983° (~76.7-78.8 m east-west, 99.7 m north-south)

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

**Zenodo**: https://doi.org/10.5281/zenodo.23465452

Download and place in `models/checkpoints/`.

### Run Inference

```python
import torch
import numpy as np
from models import VanillaUNet

# Load model
model = VanillaUNet(in_channels=6, out_channels=1)
checkpoint = torch.load('models/checkpoints/van_050105_6.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load input (6, 256, 256) - channel-first format
input_data = np.load('your_patch.npy')  # Shape: (6, 256, 256)
input_tensor = torch.from_numpy(input_data).unsqueeze(0).float()

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

De-duplicated validation pixels:

| Subset | IoU | F1 | Precision | Recall | Support |
|--------|-----|----|-----------| -------|---------|
| Training-covered | 0.61 | 0.76 | 0.90 | 0.65 | 6,444 |
| Training-uncovered | 0.11 | 0.19 | 0.58 | 0.12 | 365 |
| All validation | 0.58 | 0.74 | 0.90 | 0.62 | 6,809 |

**Key Finding**: Performance in training-uncovered pixels (IoU=0.11) is substantially lower than training-covered areas (IoU=0.61).

## Repository Structure

```
SAR-Flood-Surrogate/
├── models/
│   ├── vanilla_unet.py         # Vanilla U-Net architecture
│   ├── resnet34_unet.py        # ResNet34 U-Net architecture
│   └── checkpoints/            # Pre-trained weights (download from Zenodo)
│
├── data/
│   ├── patch_assignments.csv   # Training/validation split (1,267 / 80)
│   └── README.md
│
├── results/
│   ├── evaluation/             # Evaluation metrics (Tables 5, S1-S8)
│   └── supplementary_analysis/ # Coverage and bootstrap results
│
├── docs/
│   └── figures/                # Supplementary figures S1-S4
│
├── utils/
│   ├── make_spatial_split.py   # Generate spatial block split
│   └── run_supplementary_analysis.py
│
└── scripts/
    ├── train.py                # Training script
    ├── evaluate.py             # Evaluation script
    └── infer.py                # Inference script
```

## Training Configuration

All models were trained with (see Text S4 and Table S7 for complete specification):

- **Optimizer**: Adam
- **Learning rate**: 3×10⁻⁴
- **Learning rate schedule**: CosineAnnealingLR
- **Batch size**: 
  - 6-channel: 135
  - 7-channel: 70
- **Maximum epochs**: 30,000
- **Early stopping patience**: 1,000 epochs
- **Loss function**: 0.5 × BCE + 0.5 × Dice
- **Positive class weight**: 4.0 (in BCE term)
- **Weight decay**: 1×10⁻⁴
- **Training patches**: 1,267
- **Validation patches**: 80
- **Split strategy**: Spatial block constraint (not subbasin-based)

## Input Data Sources

### Physical Predictors

**Precipitation**:
- Source: GSMaP v6
- URL: https://sharaku.eorc.jaxa.jp/GSMaP/
- Resolution: 0.1°
- Temporal: Hourly

**Soil Moisture**:
- Source: SMAP Level-4 SPL4SMGP v007
- URL: https://nsidc.org/data/spl4smgp/
- Resolution: 9 km
- Temporal: 3-hourly

**Terrain**:
- Source: SRTM GL1 v003
- URL: https://lpdaac.usgs.gov/products/srtmgl1v003/
- Resolution: 30 m (resampled to 0.0008983°)

### SAR Flood Labels

- **Source**: Sentinel-1 C-band SAR
- **Repository**: Copernicus Data Space
- **URL**: https://dataspace.copernicus.eu/
- **Period**: July 28 - August 23, 2023
- **Processing**: GNDPI + texture + random forest (Zhu et al., 2026)
- **Labeling**: Union of all detection instances

**W₀ (Initial Water)**: Binary mask from single pre-event Sentinel-1 scene classified by same method, thresholded at 0.5.

## Input Channels

### 6-channel baseline (Vanilla and ResNet34)

1. **DEM** - Elevation (m)
2. **Slope** - Terrain gradient (degrees)
3. **Event Rainfall** - Cumulative rainfall during July 29 - August 2, 2023 (mm)
4. **W₀** - Pre-event water body binary mask
5. **Soil Moisture** - Pre-event average (m³/m³)
6. **API** - 15-day antecedent precipitation index

### 7-channel variant (Vanilla only)

Baseline 6 channels + **R1d** (1-day antecedent rainfall)

**Note**: Adding R1d degraded performance. P3d was tested with ResNet34 under loose split, not 8-channel Vanilla.

## Data Format

**Input patches**: Stored in channel-first format (C, H, W)
- Shape: (6, 256, 256) or (7, 256, 256)
- Data type: float32

**Label patches**:
- Shape: (256, 256)
- Data type: uint8 (0 or 1)

## Reproduce Results

### Reproduce Table 5

```bash
python scripts/evaluate.py \
    --checkpoint models/checkpoints/van_050105_6.pth \
    --data_split data/patch_assignments.csv \
    --output results/table5_reproduction.csv
```

### Reproduce Table S1 (Coverage Analysis)

```bash
python utils/run_supplementary_analysis.py \
    --mode coverage \
    --output results/supplementary_analysis/
```

## Code and Data Availability

**Code**: https://github.com/kaedez-hydro/SAR-Flood-Surrogate

**Model Checkpoints and Data**: https://doi.org/10.5281/zenodo.23465452
- Pre-trained models (330 MB)
- Patch assignment file
- Validation predictions

**Source Data**:
- GSMaP v6: https://sharaku.eorc.jaxa.jp/GSMaP/
- SMAP SPL4SMGP: https://nsidc.org/data/spl4smgp/
- SRTM GL1: https://lpdaac.usgs.gov/products/srtmgl1v003/
- Sentinel-1: https://dataspace.copernicus.eu/

## Citation

```bibtex
@article{flood2025sar,
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[Authors from manuscript]},
  journal={Under Review},
  year={2025},
  doi={10.5281/zenodo.23465452}
}
```

## License

- **Code**: MIT License
- **Model weights and processed data**: CC BY 4.0

## Contact

For questions:
- GitHub Issues: https://github.com/kaedez-hydro/SAR-Flood-Surrogate/issues
- Email: [From manuscript]

## Acknowledgments

- Sentinel-1 SAR: ESA Copernicus Programme
- GSMaP: JAXA EORC
- SMAP: NASA NSIDC
- SRTM: NASA/USGS LPDAAC
