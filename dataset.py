import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


class RealSRDataset(Dataset):
    def __init__(self, root_dir="../data", scale=2, crop_size=192, split="Train"):
        super().__init__()
        self.scale = scale
        self.crop_size = crop_size
        self.lr_paths = []
        self.hr_paths = []

        cameras = ["Canon", "Nikon"]

        for cam in cameras:
            lr_dir = os.path.join(root_dir, cam, split, str(scale))
            hr_dir = os.path.join(root_dir, cam, split, str(scale))

            hr_files = glob(os.path.join(hr_dir, "*_HR.png"))
            for hr_path in hr_files:
                base = os.path.basename(hr_path).replace("_HR.png", "")
                lr_path = os.path.join(lr_dir, f"{base}_LR{scale}.png")

                if os.path.exists(lr_path):
                    self.hr_paths.append(hr_path)
                    self.lr_paths.append(lr_path)

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        lr = Image.open(self.lr_paths[idx]).convert("RGB")

        w, h = lr.size
        crop = self.crop_size

        left = (w - crop) // 2
        top = (h - crop) // 2

        lr = lr.crop((left, top, left + crop, top + crop))
        hr = hr.crop(
            (
                left * self.scale,
                top * self.scale,
                left * self.scale + crop * self.scale,
                top * self.scale + crop * self.scale,
            )
        )

        return self.to_tensor(lr), self.to_tensor(hr)
