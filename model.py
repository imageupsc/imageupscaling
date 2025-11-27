import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class EDSR(nn.Module):
    """
    EDSR model
    """

    def __init__(self, scale=2, num_blocks=8, channels=64):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(3, channels, 3, padding=1)

        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
        )

        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        res = self.body_conv(res)

        x = x_head + res

        x = self.upsample(x)
        x = self.tail(x)
        return x
