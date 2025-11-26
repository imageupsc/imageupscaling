import torch
from torch.utils.data import DataLoader
from dataset import RealSRDataset


def main():
    dataset = RealSRDataset(root_dir="../data/", scale=2, split="Train")
    print(f"Количество пар LR/HR: {len(dataset)}")

    for i in range(3):
        lr, hr = dataset[i]
        print(f"Пример {i}:")
        print(
            f"  LR shape: {lr.shape}, min: {lr.min().item():.3f}, max: {lr.max().item():.3f}"
        )
        print(
            f"  HR shape: {hr.shape}, min: {hr.min().item():.3f}, max: {hr.max().item():.3f}"
        )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_lr, batch_hr = next(iter(loader))
    print("\nПроверка батча:")
    print(f"  LR batch shape: {batch_lr.shape}")
    print(f"  HR batch shape: {batch_hr.shape}")


if __name__ == "__main__":
    main()
