import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN model
    """

    def __init__(self, scale=2):
        super(SRCNN, self).__init__()
        self.scale = scale
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(
                x, scale_factor=self.scale, mode="bicubic", align_corners=False
            )
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


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
        self.tail = nn.Conv2d(channels, 3 * (scale**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = self.pixel_shuffle(x)
        return x
