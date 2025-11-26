import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


class RealSRDataset(Dataset):
    def __init__(self, root_dir, scale=2, crop_size=96, is_train=True):
        super().__init__()

        self.scale = scale
        self.crop_size = crop_size
        self.is_train = is_train

        self.lr_dir = os.path.join(root_dir, str(scale))
        self.hr_dir = os.path.join(root_dir.replace("Train", "Train"), str(scale))

        self.images = sorted(os.listdir(self.lr_dir))

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def random_crop(self, lr, hr):
        w, h = lr.size

        lr_x = random.randint(0, w - self.crop_size)
        lr_y = random.randint(0, h - self.crop_size)

        lr_crop = lr.crop((lr_x, lr_y, lr_x + self.crop_size, lr_y + self.crop_size))

        hr_crop = hr.crop(
            (
                lr_x * self.scale,
                lr_y * self.scale,
                (lr_x + self.crop_size) * self.scale,
                (lr_y + self.crop_size) * self.scale,
            )
        )

        return lr_crop, hr_crop
