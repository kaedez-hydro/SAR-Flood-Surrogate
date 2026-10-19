# Model Checkpoints

This directory contains pre-trained model checkpoints for the SAR-Flood-Surrogate project.

## Available Models

### 1. Vanilla U-Net (6 channels) - **Recommended**

**File**: `van_050105_6.pth`  
**Size**: 118.5 MB  
**Performance** (80 validation patches):
- IoU: 0.464
- F1: 0.634
- Precision: 0.817
- Recall: 0.517

**Training Details**:
- Training date: May 1, 2024
- Training patches: 1,267
- Validation patches: 80
- Epochs: 45 (best checkpoint)
- Training loss: 0.28
- Validation loss: 0.332

**Input Channels**:
1. DEM (elevation)
2. Slope (gradient)
3. Event Rainfall (cumulative)
4. Initial Water (pre-event)
5. Soil Moisture
6. API (15-day antecedent index)

---

### 2. ResNet34 U-Net (6 channels)

**File**: `res_050105_6.pth`  
**Size**: 93.2 MB  
**Performance** (80 validation patches):
- IoU: 0.400
- F1: 0.572
- Precision: 0.745
- Recall: 0.464

**Training Details**:
- Training date: May 1, 2024
- Training patches: 1,267
- Validation patches: 80
- Architecture: ResNet34 encoder + U-Net decoder

**Input Channels**: Same as Vanilla 6ch

---

### 3. Vanilla U-Net (7 channels)

**File**: `van_050105_7.pth`  
**Size**: 118.5 MB  
**Performance** (80 validation patches):
- IoU: 0.364
- F1: 0.533
- Precision: 0.644
- Recall: 0.455

**Training Details**:
- Training date: May 1, 2024
- Additional 7th channel: R1d (1-day antecedent rainfall)
- **Note**: Adding R1d degraded performance compared to 6-channel baseline

---

## Download Links

**Due to file size, model checkpoints are not included in this Git repository.**

### Option 1: Zenodo (Recommended)

- **DOI**: [To be assigned]
- **URL**: [To be provided]
- **All models**: ~310 MB total

### Option 2: Direct Download

- **Vanilla 6ch**: [Direct link]
- **ResNet34 6ch**: [Direct link]
- **Vanilla 7ch**: [Direct link]

### Option 3: Git LFS (Advanced)

If you have Git LFS installed:

```bash
git lfs install
git lfs pull
```

---

## Usage

### Load Pre-trained Model

```python
import torch
from code.train.train_model import UNet

# Initialize model
model = UNet(in_channels=6, out_channels=1)

# Load checkpoint
checkpoint = torch.load('models/checkpoints/van_050105_6.pth', 
                       map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")
```

### Run Inference

```python
import numpy as np

# Prepare input (256×256×6)
input_data = np.load('your_patch.npy')  # Shape: (256, 256, 6)
input_tensor = torch.from_numpy(input_data).float()
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 6, 256, 256)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    pred_binary = (prediction > 0.5).cpu().numpy()[0, 0]

print(f"Predicted flood pixels: {np.sum(pred_binary)}")
```

---

## Model Architecture

### Vanilla U-Net

**Encoder**:
```
Input (256, 256, C)
↓ Conv(64) + ReLU + Conv(64) + ReLU
↓ MaxPool(2×2)
↓ Conv(128) + ReLU + Conv(128) + ReLU
↓ MaxPool(2×2)
↓ Conv(256) + ReLU + Conv(256) + ReLU
↓ MaxPool(2×2)
↓ Conv(512) + ReLU + Conv(512) + ReLU
↓ MaxPool(2×2)
Bottleneck: Conv(1024) + ReLU + Conv(1024) + ReLU
```

**Decoder**:
```
↑ UpConv(512) + Concat + Conv(512) + ReLU
↑ UpConv(256) + Concat + Conv(256) + ReLU
↑ UpConv(128) + Concat + Conv(128) + ReLU
↑ UpConv(64) + Concat + Conv(64) + ReLU
Output: Conv(1) + Sigmoid → (256, 256, 1)
```

**Parameters**: ~31 million

---

### ResNet34 U-Net

**Encoder**: ResNet34 backbone (pretrained on ImageNet)

**Decoder**: Similar to Vanilla U-Net with skip connections

**Parameters**: ~25 million

---

## Training Configuration

All models were trained with:

- **Optimizer**: Adam
- **Learning rate**: 1e-4 (initial)
- **LR schedule**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Batch size**: 16
- **Loss function**: BCE + Dice (equal weight)
- **Epochs**: 200 (with early stopping, patience=20)
- **Data augmentation**:
  - Horizontal flip (p=0.5)
  - Vertical flip (p=0.5)
  - Random rotation ±15° (p=0.5)
  - Gaussian noise σ=0.01 (p=0.3)

**Hardware**: NVIDIA A100 40GB GPU

---

## Checkpoint Contents

Each `.pth` file contains:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': 45,
    'train_loss': 0.28,
    'val_loss': 0.332,
    # Additional metadata may vary
}
```

To load only the model weights:

```python
model.load_state_dict(torch.load('checkpoint.pth'))
```

---

## Model Selection Guide

**Use Vanilla 6ch (`van_050105_6.pth`) if**:
- You want the best overall performance (IoU=0.464)
- You have standard input data (6 channels)
- Baseline model is sufficient

**Use ResNet34 6ch (`res_050105_6.pth`) if**:
- You need a lighter model (fewer parameters)
- You have pretrained feature extraction needs
- Performance trade-off is acceptable

**Use Vanilla 7ch (`van_050105_7.pth`) if**:
- You are experimenting with additional inputs
- Note: This model performed worse in our tests

---

## Validation Performance Details

**Confusion Matrix** (Vanilla 6ch, aggregated over 80 patches):

```
                Predicted Positive    Predicted Negative
Actual Positive        17,930                16,727
Actual Negative         4,022             5,204,201
```

**Derived Metrics**:
- True Positives: 17,930
- False Positives: 4,022
- True Negatives: 5,204,201
- False Negatives: 16,727
- Accuracy: 99.60%

---

## Fine-tuning

To fine-tune on your own data:

```python
# Load pre-trained model
model = UNet(in_channels=6, out_channels=1)
model.load_state_dict(torch.load('van_050105_6.pth'))

# Freeze encoder layers (optional)
for param in model.encoder.parameters():
    param.requires_grad = False

# Train only decoder
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5  # Lower learning rate for fine-tuning
)

# Continue training...
```

---

## Citation

If you use these pre-trained models, please cite:

```bibtex
@article{yourname2025sar,
  title={SAR-constrained flood inundation surrogate modelling with physical predictors},
  author={[Your Names]},
  journal={Under Review},
  year={2025}
}
```

---

## License

Models are released under MIT License (see repository LICENSE file).

---

## Contact

For questions about models or download issues:
- Email: [your.email@university.edu]
- GitHub Issues: [repository URL]/issues

---

**Note**: Place downloaded `.pth` files directly in this `models/checkpoints/` directory.
