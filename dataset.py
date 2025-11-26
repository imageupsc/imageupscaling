import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class RealSRDataset(Dataset):
    """
    RealSR V3 Dataset
    LR/HR images in folders like:
    ../data/Canon/Train/2/*.png
    """

    def __init__(self, root_dir="../data/", scale=4, crop_size=192, split="Train"):
        super().__init__()
        self.scale = scale
        self.crop_size = crop_size
        self.lr_paths = []
        self.hr_paths = []

        cameras = ["Canon", "Nikon"]
        scales = ["2", "3", "4"]

        for cam in cameras:
            for sc in scales:
                lr_dir = os.path.join(root_dir, cam, split, sc)
                hr_dir = os.path.join(root_dir, cam, split, sc)

                for hr_path in glob(os.path.join(hr_dir, "*_HR.png")):
                    base_name = os.path.basename(hr_path).replace("_HR.png", "")
                    lr_path = os.path.join(lr_dir, f"{base_name}_LR{sc}.png")
                    if os.path.exists(lr_path):
                        self.lr_paths.append(lr_path)
                        self.hr_paths.append(hr_path)

        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx]).convert("RGB")
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")

        i, j, h, w = T.RandomCrop.get_params(
            lr_img, output_size=(self.crop_size, self.crop_size)
        )

        lr_img = T.functional.crop(lr_img, i, j, h, w)
        hr_img = T.functional.crop(
            hr_img, i * self.scale, j * self.scale, h * self.scale, w * self.scale
        )

        lr_img = self.transform(lr_img)
        hr_img = self.transform(hr_img)

        return lr_img, hr_img
