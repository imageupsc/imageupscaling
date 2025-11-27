# train_edsr.py
import os
from math import log10
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RealSRDataset
from model import EDSR


def ssim(img1, img2, window_size=11, size_average=True):
    """
    SSIM metric
    """
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.avg_pool2d(img1, window_size, 1, 0)
    mu2 = F.avg_pool2d(img2, window_size, 1, 0)

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, 0) - mu1.pow(2)
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, 0) - mu2.pow(2)
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, 0) - (mu1 * mu2)

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean() if size_average else ssim_map


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def calc_psnr(sr, hr):
    """
    PSNR metric
    """
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scale = 2
crop_size = 192
batch_size = 8
epochs = 70
lr = 1e-4

checkpoint_dir = "checkpoints_edsr"
os.makedirs(checkpoint_dir, exist_ok=True)


train_dataset = RealSRDataset(
    root_dir="../data", scale=scale, split="Train", crop_size=crop_size
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = RealSRDataset(
    root_dir="../data", scale=scale, split="Test", crop_size=crop_size
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# model, loss, optimizer
model = EDSR(scale=scale, num_blocks=8, channels=64).to(device)
criterion = CharbonnierLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)


best_psnr = 0


for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for lr_imgs, hr_imgs in pbar:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        psnr_val = calc_psnr(sr_imgs, hr_imgs)
        ssim_val = ssim(sr_imgs, hr_imgs)

        total_loss += loss.item()
        total_psnr += psnr_val
        total_ssim += ssim_val.item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "psnr": f"{psnr_val:.2f}",
                "ssim": f"{ssim_val.item():.4f}",
            }
        )

    avg_loss = total_loss / len(train_loader)
    avg_psnr = total_psnr / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)

    print(
        f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}"
    )

    model.eval()
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            val_psnr += calc_psnr(sr_imgs, hr_imgs)
            val_ssim += ssim(sr_imgs, hr_imgs).item()

    avg_val_psnr = val_psnr / len(test_loader)
    avg_val_ssim = val_ssim / len(test_loader)

    print(f"Validation: PSNR={avg_val_psnr:.2f}, SSIM={avg_val_ssim:.4f}")

    scheduler.step(avg_val_psnr)

    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        print(f"New BEST model saved! PSNR={avg_val_psnr:.2f}\n")

print("Training finished!")
