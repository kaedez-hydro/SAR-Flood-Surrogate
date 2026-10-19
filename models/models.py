from typing import Iterable

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VanillaUNet(nn.Module):
    """Classic U-Net without pretrained encoder."""

    def __init__(self, n_channels: int = 8, n_classes: int = 1, base_c: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.outc = OutConv(base_c, n_classes)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def _build_resnet34_unet(n_channels: int, n_classes: int) -> nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=n_channels,
        classes=n_classes,
        activation=None,
    )


def _build_deeplabv3plus(n_channels: int, n_classes: int) -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=n_channels,
        classes=n_classes,
        activation=None,
    )


def build_model(model_name: str, n_channels: int, n_classes: int = 1,
                dropout: float = 0.0) -> nn.Module:
    """
    Unified model factory.

    Supported model names:
    - resnet34_unet
    - vanilla_unet
    - deeplabv3plus
    """
    key = model_name.strip().lower()

    if key == "resnet34_unet":
        return _build_resnet34_unet(n_channels=n_channels, n_classes=n_classes)
    if key == "vanilla_unet":
        return VanillaUNet(n_channels=n_channels, n_classes=n_classes,
                           dropout=dropout)
    if key == "deeplabv3plus":
        return _build_deeplabv3plus(n_channels=n_channels, n_classes=n_classes)

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        f"Expected one of: {', '.join(list_supported_models())}."
    )


def list_supported_models() -> Iterable[str]:
    return ("resnet34_unet", "vanilla_unet", "deeplabv3plus")


def SimpleUNet(n_channels: int = 8, n_classes: int = 1) -> nn.Module:
    """
    Backward-compatible alias for existing training code.
    Maps to resnet34_unet.
    """
    return build_model("resnet34_unet", n_channels=n_channels, n_classes=n_classes)
