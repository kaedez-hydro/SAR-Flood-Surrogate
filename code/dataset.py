import os
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm


class FloodDataset(Dataset):
    def __init__(self, data_dir, transform=False, cache_memory=True):
        self.input_dir = os.path.join(data_dir, "inputs")
        self.label_dir = os.path.join(data_dir, "labels")

        self.files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.npy')])
        self.transform = bool(transform)
        self.cache_memory = bool(cache_memory)

        self.cached_data = []
        if self.cache_memory:
            print(f"[Cache] Loading {len(self.files)} samples into memory...")
            for fname in tqdm(self.files, desc="Loading Data"):
                img = np.load(os.path.join(self.input_dir, fname)).astype(np.float32)
                mask = np.load(os.path.join(self.label_dir, fname)).astype(np.float32)
                self.cached_data.append((img, mask))
            print("[Cache] Done.")

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.GaussNoise(var_limit=(0, 0.01), p=0.5),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                            fill_value=0.0, mask_fill_value=0.0, p=0.1),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_memory:
            img, mask = self.cached_data[idx]
            img = img.copy()
            mask = mask.copy()
        else:
            fname = self.files[idx]
            img = np.load(os.path.join(self.input_dir, fname)).astype(np.float32)
            mask = np.load(os.path.join(self.label_dir, fname)).astype(np.float32)

        if self.transform:
            img_hwc = np.transpose(img, (1, 2, 0))
            augmented = self.aug(image=img_hwc, mask=mask)
            img = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return img_tensor, mask_tensor
