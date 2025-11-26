from dataset import RealSRDataset
from torch.utils.data import DataLoader

ds = RealSRDataset("../data/Train", scale=2, crop_size=96)
dl = DataLoader(ds, batch_size=4, shuffle=True)

lr, hr = next(iter(dl))

print("LR batch:", lr.shape)
print("HR batch:", hr.shape)
