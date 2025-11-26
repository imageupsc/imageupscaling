import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from dataset import RealSRDataset
from model import SRModel


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = RealSRDataset(
        root_dir="data/Train", scale=2, crop_size=96, is_train=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    )

    # --- MODEL ---
    model = SRModel(scale=2).to(device)

    # --- LOSS ---
    criterion = nn.L1Loss()

    # --- OPTIMIZER ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- TRAIN LOOP ---
    epochs = 50

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for lr, hr in pbar:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)

            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_psnr += calc_psnr(sr.detach(), hr).item()

            pbar.set_postfix(
                {
                    "loss": f"{epoch_loss / len(train_loader):.4f}",
                    "psnr": f"{epoch_psnr / len(train_loader):.2f}",
                }
            )

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")


if __name__ == "__main__":
    train()
