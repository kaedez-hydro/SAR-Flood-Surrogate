# Model Checkpoints

Pre-trained model weights for the SAR-Flood-Surrogate project.

## Download

**All checkpoints (~330 MB total) are available from Zenodo:**

[![DOI](https://zenodo.org/badge/DOI/[TO_BE_ADDED].svg)](https://doi.org/[TO_BE_ADDED])

Download and place `.pth` files in this directory.

---

## Available Models

### 1. Vanilla U-Net (6 channels) - Recommended

**File**: `van_050105_6.pth` (~110 MB)

**Performance** (80 validation patches, Table 5):
- IoU: 0.464
- F1: 0.634
- Precision: 0.817
- Recall: 0.517

**Input Channels**:
1. DEM
2. Slope
3. Event Rainfall
4. Initial Water
5. Soil Moisture
6. API (15-day)

---

### 2. ResNet34 U-Net (6 channels)

**File**: `res_050105_6.pth` (~93 MB)

**Performance** (80 validation patches, Table 5):
- IoU: 0.400
- F1: 0.572
- Precision: 0.745
- Recall: 0.464

**Input Channels**: Same as Vanilla 6ch

---

### 3. Vanilla U-Net (7 channels)

**File**: `van_050105_7.pth` (~110 MB)

**Performance** (80 validation patches, Table 5):
- IoU: 0.364
- F1: 0.533
- Precision: 0.644
- Recall: 0.455

**Input Channels**: 6 baseline + R1d (1-day antecedent rainfall)

**Note**: Adding R1d degraded performance compared to 6-channel baseline.

---

## Training Details

All models were trained with:

- **Optimizer**: Adam
- **Learning rate**: 3×10⁻⁴
- **Learning rate schedule**: CosineAnnealingLR
- **Batch size**: 135 (training), 70 (validation)
- **Maximum epochs**: 30,000
- **Early stopping patience**: 1,000 epochs
- **Loss function**: Weighted Binary Cross-Entropy
  - Positive class weight: 4.0
- **Weight decay**: 1×10⁻⁴
- **Training data**: 1,267 patches
- **Validation data**: 80 patches

See supplementary materials for complete hyperparameter specification.

---

## Model Architectures

### Vanilla U-Net

Standard U-Net architecture with:
- 5 encoder levels
- 4 decoder levels with skip connections
- Sigmoid activation at output

**Parameters**: ~31 million

### ResNet34 U-Net

U-Net with ResNet34 encoder:
- ResNet34 backbone as encoder
- U-Net decoder with skip connections
- Sigmoid activation at output

**Parameters**: ~25 million

---

## Usage

### Load Checkpoint

```python
import torch
from code.models import VanillaUNet

# Initialize model
model = VanillaUNet(in_channels=6, out_channels=1)

# Load checkpoint
checkpoint = torch.load('models/checkpoints/van_050105_6.pth', 
                       map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Run Inference

```python
import numpy as np

# Load input patch (256, 256, 6)
input_data = np.load('your_patch.npy')

# Convert to tensor (1, 6, 256, 256)
input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0).float()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction_prob = torch.sigmoid(output)
    prediction_binary = (prediction_prob > 0.5).cpu().numpy()[0, 0]

print(f"Predicted flood pixels: {np.sum(prediction_binary)}")
```

---

## Checkpoint Contents

Each `.pth` file contains a dictionary with:

```python
{
    'model_state_dict': <model weights>,
    # Additional metadata may be present
}
```

Load only model weights:

```python
checkpoint = torch.load('van_050105_6.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Model Selection Guide

**Use Vanilla 6ch** if:
- You want the best performance (IoU=0.464)
- Standard input data available (6 channels)
- This is the recommended baseline

**Use ResNet34 6ch** if:
- You need fewer parameters
- Comparable performance acceptable (IoU=0.400)

**Use Vanilla 7ch** for:
- Experimentation only
- Note: Performed worse than 6ch baseline

---

## Validation Performance Breakdown

### Vanilla 6ch (Recommended)

**Patch-level** (80 patches, Table 5):
- IoU: 0.464
- F1: 0.634
- Precision: 0.817
- Recall: 0.517

**De-duplicated pixels** (Table S1):
- Training-covered (6,444 positives): IoU = 0.61
- Training-uncovered (365 positives): IoU = 0.11
- All validation (6,809 positives): IoU = 0.58

**Confusion Matrix** (patch-level aggregate):
- True Positives: 17,930
- False Positives: 4,022
- True Negatives: 5,204,201
- False Negatives: 16,727

---

## Fine-tuning

To fine-tune on your own data:

```python
# Load pre-trained model
model = VanillaUNet(in_channels=6, out_channels=1)
checkpoint = torch.load('van_050105_6.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Optional: freeze encoder
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False

# Set up optimizer (lower learning rate for fine-tuning)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# Continue training...
```

---

## System Requirements

**For inference**:
- GPU: NVIDIA GPU with ≥4GB VRAM
- RAM: ≥8GB
- PyTorch: ≥1.10.0

**For training**:
- GPU: NVIDIA GPU with ≥16GB VRAM
- RAM: ≥32GB
- PyTorch: ≥1.10.0

---

## Citation

If you use these pre-trained models, please cite:

```bibtex
@article{[TO_BE_ADDED],
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[TO_BE_ADDED]},
  journal={Under Review},
  year={2025}
}
```

---

## License

Model weights are released under the MIT License.

Available for non-commercial research purposes.

---

## Contact

For questions about models:
- GitHub Issues: https://github.com/kaedez-hydro/SAR-Flood-Surrogate/issues
- Email: [TO_BE_ADDED]
