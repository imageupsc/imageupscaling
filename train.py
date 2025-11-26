import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import RealSRDataset
from model import SRCNN
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10
import os

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale = 2
batch_size = 4
epochs = 10  # increase after test
lr = 1e-4
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


train_dataset = RealSRDataset("data/Train", scale=scale, crop_size=96, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = RealSRDataset("data/Test", scale=scale, crop_size=96, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model, loss, optimizer
model = SRCNN(scale=scale).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def calc_psnr(sr, hr):
    """
    Calculating PSNR
    """
    mse = criterion(sr, hr)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())


# Training cycle
for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0
    epoch_psnr = 0

    for lr_imgs, hr_imgs in pbar:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)

        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        psnr = calc_psnr(sr_imgs, hr_imgs)

        epoch_loss += loss.item()
        epoch_psnr += psnr

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{psnr:.2f}"})

    avg_loss = epoch_loss / len(train_loader)
    avg_psnr = epoch_psnr / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.2f} dB")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    model.eval()
    with torch.no_grad():
        for lr_img, hr_img in test_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr_img = model(lr_img)

            sr_np = sr_img[0].cpu().permute(1, 2, 0).numpy()
            hr_np = hr_img[0].cpu().permute(1, 2, 0).numpy()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hr_np)
            plt.title("HR")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(sr_np)
            plt.title("SR")
            plt.axis("off")
            plt.show()
            break

print("Training finished!")
