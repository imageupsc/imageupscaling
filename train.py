# train.py
import os
from math import log10
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RealSRDataset
from model import build_model


def ssim(img1, img2, window_size=11, size_average=True):
    """Compute SSIM for batch of images"""
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.avg_pool2d(img1, window_size, 1, 0)
    mu2 = F.avg_pool2d(img2, window_size, 1, 0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, 0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, 0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, 0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


def calc_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())


# properties
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale = 2
crop_size = 96
batch_size = 1
epochs = 5
lr = 1e-4
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


train_dataset = RealSRDataset(root_dir="../data", scale=scale, split="Train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = RealSRDataset(root_dir="../data", scale=scale, split="Test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# model, loss, optimizer
model = build_model(scale=scale).to(device)
criterion = CharbonnierLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

best_psnr = 0

for epoch in range(epochs):
    model.train()
    epoch_loss, epoch_psnr, epoch_ssim = 0, 0, 0

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

        epoch_loss += loss.item()
        epoch_psnr += psnr_val
        epoch_ssim += ssim_val.item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "psnr": f"{psnr_val:.2f}",
                "ssim": f"{ssim_val.item():.4f}",
            }
        )

    scheduler.step()

    avg_loss = epoch_loss / len(train_loader)
    avg_psnr = epoch_psnr / len(train_loader)
    avg_ssim = epoch_ssim / len(train_loader)

    print(
        f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}"
    )

    # validation
    model.eval()
    test_psnr, test_ssim = 0, 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            test_psnr += calc_psnr(sr_imgs, hr_imgs)
            test_ssim += ssim(sr_imgs, hr_imgs).item()

    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)
    print(f"Test PSNR: {avg_test_psnr:.2f}, Test SSIM: {avg_test_ssim:.4f}")

    # save best model
    if avg_test_psnr > best_psnr:
        best_psnr = avg_test_psnr
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        print(f"--> New best model saved at epoch {epoch+1} (PSNR={best_psnr:.2f})")

print("Training finished!")
