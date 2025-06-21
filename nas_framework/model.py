import torch
import torch.nn as nn
from typing import List

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x

class PoolBlock(nn.Module):
    def __init__(self, mode: str = 'max'):
        super().__init__()
        if mode == 'avg':
            self.pool = nn.AvgPool2d(2)
        else:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)

BLOCK_TYPES = {
    0: ConvBlock,
    1: TransformerBlock,
    2: PoolBlock,
}

def build_model(block_ids: List[int], in_channels: int = 3, num_classes: int = 100):
    layers = []
    channels = in_channels
    for bid in block_ids:
        if bid == 0:
            layers.append(ConvBlock(channels))
        elif bid == 1:
            layers.append(TransformerBlock(channels))
        elif bid == 2:
            layers.append(PoolBlock())
        channels = 64  # keep constant for simplicity
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(channels, num_classes))
    return nn.Sequential(*layers)
