# SAR-Constrained Flood Inundation Surrogate Modelling

Deep learning surrogate model for rapid flood extent mapping using SAR observations and physical predictors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-red.svg)](https://pytorch.org/)

## 📄 Paper

**Title**: SAR-constrained flood inundation surrogate modelling with physical predictors

**Study Event**: July 2023 Daqinghe River flood, North China Plain

**Status**: Under review

## 🎯 Overview

This repository contains code, data, and results for training and evaluating deep learning surrogate models that predict flood inundation extent from terrain, meteorological, and hydrological inputs, supervised by SAR satellite observations.

### Key Features

- **Multi-architecture support**: Vanilla U-Net and ResNet34 U-Net
- **Multi-channel inputs**: DEM, slope, rainfall, soil moisture, API, etc.
- **Spatial block validation**: Conservative assessment with geographic holdout
- **Full-basin reconstruction**: Sliding-window inference for complete domain
- **Comprehensive evaluation**: Patch-level and subset-based performance analysis

### Study Area

- **Location**: Daqinghe River Basin, North China Plain
- **Event**: July 2023 extreme flood
- **Domain**: 2,176 × 4,672 pixels (100m resolution)
- **Total area**: ~200 km × 460 km

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: ≥8GB VRAM for inference, ≥40GB for training)
- 16GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SAR-Flood-Surrogate.git
cd SAR-Flood-Surrogate

# Create conda environment
conda create -n flood python=3.8
conda activate flood

# Install dependencies
pip install -r requirements.txt
```

### Download Pre-trained Models

Pre-trained model checkpoints are available:

- **Vanilla U-Net (6 channels)**: [Download link] (118.5 MB)
- **ResNet34 U-Net (6 channels)**: [Download link] (93.2 MB)
- **Vanilla U-Net (7 channels)**: [Download link] (118.5 MB)

Place downloaded checkpoints in `models/checkpoints/`.

### Quick Inference Example

```python
import torch
import numpy as np
from code.train.train_model import UNet

# Load model
model = UNet(in_channels=6, out_channels=1)
checkpoint = torch.load('models/checkpoints/van_050105_6.pth')
model.load_state_dict(checkpoint)
model.eval()

# Prepare input (256×256×6)
# Channels: [DEM, Slope, Rainfall, Initial Water, Soil Moisture, API]
input_data = np.load('your_input_patch.npy')
input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)

# Run inference
with torch.no_grad():
    prediction = model(input_tensor)
    pred_binary = (prediction > 0.5).numpy()
```

## 📊 Results

### Table 4: Model Performance on 80 Validation Patches

| Model | Channels | IoU | F1 | Precision | Recall | Area Bias |
|-------|----------|-----|----|-----------| -------|-----------|
| ResNet34 U-Net | 6 | 0.400 | 0.572 | 0.745 | 0.464 | -0.377 |
| **Vanilla U-Net** | **6** | **0.464** | **0.634** | **0.817** | **0.517** | **-0.367** |
| Vanilla U-Net | 7 | 0.364 | 0.533 | 0.644 | 0.455 | -0.294 |

**Key Findings**:
- Vanilla U-Net outperforms ResNet34 U-Net at same channel count
- Adding short-term antecedent rainfall (7th channel) degrades performance
- 6-channel configuration is optimal

### Validation Coverage Analysis

From supplementary analysis (October 2024):

- **Total validation pixels**: 1,105,920 (de-duplicated)
- **Training-covered pixels**: 790,528 (71.48%)
- **Training-uncovered pixels**: 315,392 (28.52%)

**Performance by coverage**:
- Training-covered: IoU = 0.50
- Training-uncovered: IoU = 0.14

This indicates validation primarily assesses within-domain generalization rather than cross-domain transfer.

### Full Evaluation Results

All results available in `results/`:
- `evaluation/evaluation_summary.csv`: Aggregated metrics
- `evaluation/patch_level_results.csv`: Per-patch performance
- `supplementary_analysis/`: Coverage and subset analysis

## 📁 Repository Structure

```
SAR-Flood-Surrogate/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore rules
│
├── code/
│   ├── train/
│   │   └── train_model.py     # Training script with model definitions
│   ├── inference/
│   ├── evaluation/
│   └── utils/
│
├── models/
│   ├── checkpoints/           # Pre-trained model weights (download separately)
│   │   ├── README.md         # Download instructions
│   │   └── .gitkeep
│   └── configs/              # Model configuration files
│
├── data/
│   ├── README.md             # Data description and download links
│   ├── sample_patches/       # Sample training patches (if included)
│   └── patch_info/          # Patch metadata
│
├── results/
│   ├── evaluation/          # Model evaluation results
│   └── supplementary_analysis/  # October 2024 supplementary analysis
│
├── docs/
│   ├── figures/            # Supplementary figures (S1-S4)
│   └── INSTALL.md         # Installation guide
│
├── scripts/               # Utility scripts
└── tests/                # Unit tests
```

## 🔬 Reproduce Results

### Reproduce Table 4

```bash
# Using pre-trained checkpoint
python code/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/van_050105_6.pth \
    --data path/to/validation_patches/ \
    --output results/reproduction/
```

### Train Your Own Model

```bash
python code/train/train_model.py \
    --data_dir path/to/training_data/ \
    --model vanilla_unet \
    --channels 6 \
    --batch_size 16 \
    --epochs 200 \
    --lr 1e-4 \
    --output_dir outputs/
```

Training requires:
- Training data: 1,267 patches (256×256×6)
- GPU: NVIDIA A100 40GB or equivalent
- Time: ~4-6 hours to convergence

## 📦 Data Availability

### Input Data Sources

**Satellite Imagery**:
- **SAR**: Sentinel-1 C-band (Copernicus Open Access Hub)
  - URL: https://scihub.copernicus.eu/
  - Period: July 28 - August 23, 2023
  - Product: GRD (Ground Range Detected)

**Meteorological Data**:
- **ERA5-Land**: Soil moisture, rainfall (Copernicus Climate Data Store)
  - URL: https://cds.climate.copernicus.eu/
  - Variables: Volumetric soil water, total precipitation

**Terrain Data**:
- **SRTM DEM**: 30m elevation (USGS EarthExplorer)
  - URL: https://earthexplorer.usgs.gov/
  - Resampled to 100m for this study

### Processed Training Data

Due to large file size (~60 GB), processed training patches are available on request or through external repository:

- **Zenodo**: [DOI to be assigned]
- **Contents**:
  - 1,347 patches (256×256 pixels)
  - 6-8 input channels
  - Binary flood labels from SAR
  - Patch metadata (training/validation split)

### Model Checkpoints

Pre-trained models available at:
- **Zenodo**: [DOI to be assigned]
- **Models**:
  - `van_050105_6.pth`: Vanilla U-Net, 6 channels (118.5 MB)
  - `res_050105_6.pth`: ResNet34 U-Net, 6 channels (93.2 MB)
  - `van_050105_7.pth`: Vanilla U-Net, 7 channels (118.5 MB)

## 🛠️ System Requirements

### Hardware

**For Inference**:
- GPU: NVIDIA GPU with ≥8GB VRAM
- RAM: ≥16GB
- Storage: ≥10GB

**For Training**:
- GPU: NVIDIA GPU with ≥40GB VRAM (e.g., A100)
- RAM: ≥64GB
- Storage: ≥100GB

### Software

- **OS**: Linux (Ubuntu 18.04+), Windows 10+, or macOS
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.7 or higher (for GPU acceleration)
- **PyTorch**: 1.13.0 or higher

## 📖 Documentation

- **Installation Guide**: `docs/INSTALL.md`
- **Model Architecture**: See `code/train/train_model.py`
- **Supplementary Materials**: Available in `docs/`
- **API Reference**: Docstrings in code

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{yourname2025sar,
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[Your Names]},
  journal={Under Review},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

- **Lead Author**: [your.email@university.edu]
- **Lab Website**: [https://yourlab.edu]
- **Issues**: Please use GitHub Issues for bug reports and feature requests

## 🙏 Acknowledgments

- **Data Sources**:
  - ESA Copernicus Programme (Sentinel-1 SAR)
  - ECMWF (ERA5-Land reanalysis)
  - NASA/USGS (SRTM DEM)

- **Computing Resources**:
  - OpenBayes Platform for model training

- **Supplementary Analysis**:
  - October 2024 independent verification and coverage analysis

## 📚 Related Work

- **U-Net**: Ronneberger et al. (2015) - Medical Image Segmentation
- **ResNet**: He et al. (2016) - Deep Residual Learning
- **SAR Flood Mapping**: Schumann & Moller (2015) - Microwave Remote Sensing

## 🔄 Updates

- **2025-01**: Initial release with three pre-trained models
- **2024-10**: Supplementary analysis results added
- **2024-05**: Model training completed

---

**Last Updated**: January 2025

For questions, please contact the corresponding author or open an issue on GitHub.
